import time
import numpy as np
import joblib

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =======================
# CONFIG
# =======================
BOW_READY = False
BOW_PATH = "bow_knn_jaccard.obj"
VEC_PATH = "vectorizer_knn_jaccard.obj"

NUM_PERM = 32          # try 16 / 32 / 64
TAU = 0.9              # try 0.9 then tune down (0.8, 0.7, ...)
K = 7                  # assignment uses K=7 for LSH part
MINHASH_CACHE = f"minhash_all_numperm_{NUM_PERM}.obj"


# =======================
# Load training data
# =======================
dataTrain = joblib.load("data_train.obj")

print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())


# =======================
# Build / Load Binary BoW
# =======================
print("\n--- Building/Loading Binary BoW ---")

if not BOW_READY:
    bow_vectorizer = CountVectorizer(
        binary=True,          # IMPORTANT for Jaccard / set semantics
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=1000
    )
    X = bow_vectorizer.fit_transform(dataTrain["Content"].fillna("").astype(str)).tocsr()
    joblib.dump(X, BOW_PATH)
    joblib.dump(bow_vectorizer, VEC_PATH)
else:
    bow_vectorizer = joblib.load(VEC_PATH)
    X = joblib.load(BOW_PATH).tocsr()

y = dataTrain["Label"].values

print("X:", X.shape, " nnz:", X.nnz)
print("y:", y.shape)


# =======================
# Helpers
# =======================
def minhash_from_sparse_row(row_csr, num_perm: int) -> MinHash:
    """
    Build MinHash from a 1xV CSR sparse row by treating the active feature
    indices as "tokens".
    """
    mh = MinHash(num_perm=num_perm)
    for j in row_csr.indices:
        mh.update(str(j).encode("utf8"))
    return mh


def jaccard_sparse_binary(a_csr, b_csr) -> float:
    """
    Exact Jaccard similarity between two 1xV sparse binary rows.
    """
    inter = a_csr.multiply(b_csr).nnz
    union = a_csr.nnz + b_csr.nnz - inter
    return (inter / union) if union else 0.0


def majority_vote(labels: np.ndarray):
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def predict_one_with_lsh(X_tr, y_tr, q_row, lsh: MinHashLSH, q_mh: MinHash, k: int):
    """
    LSH -> candidates -> exact Jaccard -> top-k -> majority vote.
    If no candidates, fall back to brute force within X_tr (slow, but safe).
    """
    cand_keys = lsh.query(q_mh)

    if not cand_keys:
        #cand_idx = range(X_tr.shape[0])
        return majority_vote(y_tr), 0
    else:
        cand_idx = [int(s) for s in cand_keys]  # these are indices into X_tr

    scored = []
    for idx in cand_idx:
        sim = jaccard_sparse_binary(q_row, X_tr[idx])
        scored.append((sim, idx))

    scored.sort(reverse=True, key=lambda t: t[0])
    topk_idx = [idx for _, idx in scored[:k]]

    return majority_vote(y_tr[topk_idx]), len(scored)


# =======================
# Precompute MinHashes ONCE and save
# =======================
if not joblib.os.path.exists(MINHASH_CACHE):
    print(f"\n--- Precomputing MinHash for ALL docs (num_perm={NUM_PERM}) ---")
    t0 = time.perf_counter()

    all_mh = [None] * X.shape[0]
    for i in tqdm(range(X.shape[0]), desc="MinHash(all docs)", unit="doc"):
        all_mh[i] = minhash_from_sparse_row(X[i], num_perm=NUM_PERM)

    joblib.dump(all_mh, MINHASH_CACHE)
    print(f"Saved: {MINHASH_CACHE}")
    print(f"Precompute time: {time.perf_counter() - t0:.2f}s")
else:
    print(f"\n--- Loading cached MinHashes: {MINHASH_CACHE} ---")
    all_mh = joblib.load(MINHASH_CACHE)


# =======================
# 5-fold Stratified CV using cached MinHashes
# =======================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc = []
all_true = []
all_pred = []

build_times = []
query_times = []
total_times = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
    print(f"\n=== Fold {fold} ===")

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # Build an LSH index for THIS fold's train set
    t_build = time.perf_counter()
    lsh = MinHashLSH(threshold=TAU, num_perm=NUM_PERM)

    # IMPORTANT: in this fold, we re-index train docs 0..len(tr_idx)-1
    # so we need a mapping: local_train_i -> global_i
    local_to_global = np.array(tr_idx)

    for local_i in tqdm(range(len(local_to_global)), desc="LSH insert(train)", unit="doc"):
        global_i = int(local_to_global[local_i])
        lsh.insert(str(local_i), all_mh[global_i])

    build_t = time.perf_counter() - t_build

    # Query / predict validation fold
    t_query = time.perf_counter()
    preds = []
    cand_counts = []

    for local_j in tqdm(range(len(va_idx)), desc="Predict(valid)", unit="doc"):
        global_j = int(va_idx[local_j])
        q_row = X_va[local_j]

        # Build query MinHash from cached (global) as well
        q_mh = all_mh[global_j]

        pred, ccount = predict_one_with_lsh(X_tr, y_tr, q_row, lsh, q_mh, k=K)
        preds.append(pred)
        cand_counts.append(ccount)

    query_t = time.perf_counter() - t_query
    total_t = build_t + query_t

    acc = accuracy_score(y_va, preds)

    build_times.append(build_t)
    query_times.append(query_t)
    total_times.append(total_t)

    fold_acc.append(acc)
    all_true.extend(y_va)
    all_pred.extend(preds)

    print(f"Fold {fold} accuracy: {acc:.4f}")
    print(f"Fold {fold} build time: {build_t:.2f}s | query time: {query_t:.2f}s | total: {total_t:.2f}s")
    print(f"Fold {fold} avg candidates scored/query: {float(np.mean(cand_counts)):.1f}")

print("\n=== Summary ===")
print("Mean CV accuracy:", float(np.mean(fold_acc)), "+/-", float(np.std(fold_acc)))
print("Mean build time:", float(np.mean(build_times)), "s")
print("Mean query time:", float(np.mean(query_times)), "s")
print("Mean total time:", float(np.mean(total_times)), "s")

print("\nOverall classification report (aggregated over folds):")
print(classification_report(all_true, all_pred))
print("Confusion matrix (aggregated over folds):")
print(confusion_matrix(all_true, all_pred))

