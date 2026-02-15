import time
import numpy as np
import joblib

from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------
# Load training data
# -----------------------
dataTrain = joblib.load("data_train.obj")

print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())


# -----------------------
# Build / Load Binary BoW (same as before)
# -----------------------
print("Binary BoW for Jaccard + LSH")
bowReady = False

if not bowReady:
    bow_vectorizer = CountVectorizer(
        binary=True,            # IMPORTANT for Jaccard / set semantics
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=1000       # keep small first, then scale up
    )
    X_bow = bow_vectorizer.fit_transform(dataTrain["Content"].fillna("").astype(str))
    joblib.dump(X_bow, "bow_knn_jaccard.obj")
    joblib.dump(bow_vectorizer, "vectorizer_knn_jaccard.obj")
else:
    bow_vectorizer = joblib.load("vectorizer_knn_jaccard.obj")
    X_bow = joblib.load("bow_knn_jaccard.obj")

X = X_bow.tocsr()
y = dataTrain["Label"].values

print("\n--- Shapes ---")
print("X:", X.shape)
print("y:", y.shape)


# -----------------------
# Helpers: MinHash + exact Jaccard on sparse binary
# -----------------------
def minhash_from_sparse_row(row_csr, num_perm: int) -> MinHash:
    """
    row_csr: 1xV CSR sparse row with binary values (0/1).
    We build a MinHash over the set of active feature indices.
    """
    mh = MinHash(num_perm=num_perm)
    # row_csr.indices are the column indices of nonzeros
    for j in row_csr.indices:
        mh.update(str(j).encode("utf8"))
    return mh


def jaccard_sparse_binary(a_csr, b_csr) -> float:
    """
    Exact Jaccard similarity between two 1xV sparse binary rows.
    sim = |A∩B| / |A∪B|
    """
    # intersection count
    inter = a_csr.multiply(b_csr).nnz
    # union count
    union = a_csr.nnz + b_csr.nnz - inter
    return (inter / union) if union else 0.0


def knn_predict_lsh(
    X_train, y_train, X_query,
    lsh: MinHashLSH, train_minhashes,
    k: int
):
    """
    Predict label for a single query row using:
    LSH -> candidate set -> exact Jaccard -> top-k -> majority vote.
    """
    # query candidates using MinHash
    q_mh = minhash_from_sparse_row(X_query, num_perm=train_minhashes[0].num_perm)
    cand_keys = lsh.query(q_mh)

    if not cand_keys:
        # No candidates: fallback to brute within train (slow but safe)
        cand_idx = range(X_train.shape[0])
    else:
        cand_idx = [int(c) for c in cand_keys]

    # score candidates with exact Jaccard
    scored = []
    for idx in cand_idx:
        sim = jaccard_sparse_binary(X_query, X_train[idx])
        scored.append((sim, idx))

    # take top-k most similar
    scored.sort(reverse=True, key=lambda t: t[0])
    topk = [idx for _, idx in scored[:k]]

    # majority vote
    labels = y_train[topk]
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


# -----------------------
# 5-fold Stratified CV + LSH timing/accuracy
# -----------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

K = 7            # assignment uses K=7 for LSH part
tau = 0.9        # starting threshold
num_perm = 32    # try 16, 32, 64 in experiments

fold_acc = []
all_true = []
all_pred = []

build_times = []
query_times = []
total_times = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # ---- Build LSH on TRAIN fold ----
    t0 = time.perf_counter()

    lsh = MinHashLSH(threshold=tau, num_perm=num_perm)

    # store train minhashes so we know num_perm and avoid rebuilding if you extend this later
    train_minhashes = [None] * X_tr.shape[0]

    for i in range(X_tr.shape[0]):
        mh = minhash_from_sparse_row(X_tr[i], num_perm=num_perm)
        train_minhashes[i] = mh
        lsh.insert(str(i), mh)

    build_t = time.perf_counter() - t0

    # ---- Query / predict VALIDATION fold ----
    t1 = time.perf_counter()
    preds = []
    for i in range(X_va.shape[0]):
        pred = knn_predict_lsh(
            X_train=X_tr,
            y_train=y_tr,
            X_query=X_va[i],
            lsh=lsh,
            train_minhashes=train_minhashes,
            k=K
        )
        preds.append(pred)

    query_t = time.perf_counter() - t1
    total_t = build_t + query_t

    acc = accuracy_score(y_va, preds)

    build_times.append(build_t)
    query_times.append(query_t)
    total_times.append(total_t)

    fold_acc.append(acc)
    all_true.extend(y_va)
    all_pred.extend(preds)

    print(f"Fold {fold} accuracy: {acc:.4f} | build={build_t:.2f}s query={query_t:.2f}s total={total_t:.2f}s")

print("\nMean CV accuracy:", float(np.mean(fold_acc)), "+/-", float(np.std(fold_acc)))
print("Mean build time:", float(np.mean(build_times)), "s")
print("Mean query time:", float(np.mean(query_times)), "s")
print("Mean total time:", float(np.mean(total_times)), "s")

print("\nOverall classification report (aggregated over folds):")
print(classification_report(all_true, all_pred))
print("Confusion matrix (aggregated over folds):")
print(confusion_matrix(all_true, all_pred))

