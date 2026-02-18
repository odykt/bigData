import os
import time
import numpy as np
import pandas as pd
import joblib

from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================
# CONFIG
# =========================
DATA_PATH = "part1/joblibCache/dataTrain_cleaned.joblib"

K = 7
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Part 2 params to sweep
TAU_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_PERM_VALUES = [16, 32, 64]

# Optional speed switch for debugging:
MAX_QUERIES = None  # e.g. 2000, or None for all test docs


# =========================
# Load data
# =========================
data_train = joblib.load(DATA_PATH)
X_text = data_train["Content"].fillna("").astype(str)
y = data_train["Label"].values


# =========================
# Train/Test split
# =========================
X_text_tr, X_text_te, y_tr, y_te = train_test_split(
    X_text,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# =========================
# Build Binary BoW (fit on train only)
# =========================
print("\n--- Building Binary BoW (fit on train only) ---")
bow_vectorizer = CountVectorizer(
    binary=True,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.8,
    max_features=1000,
)

X_tr = bow_vectorizer.fit_transform(X_text_tr).tocsr()
X_te = bow_vectorizer.transform(X_text_te).tocsr()

print("X_tr:", X_tr.shape, "nnz:", X_tr.nnz)
print("X_te:", X_te.shape, "nnz:", X_te.nnz)
print("y_tr:", y_tr.shape, "y_te:", y_te.shape)


# =========================
# Helpers
# =========================
def majority_vote(labels: np.ndarray):
    vals, cnts = np.unique(labels, return_counts=True)
    return vals[np.argmax(cnts)]


def minhash_from_sparse_row(row_csr, num_perm: int) -> MinHash:
    """MinHash over the set of active feature indices in a binary CSR row."""
    mh = MinHash(num_perm=num_perm)
    for j in row_csr.indices:
        mh.update(str(j).encode("utf8"))
    return mh


def topk_true_jaccard_indices(X_tr_local, q_row, k: int) -> np.ndarray:
    """
    TRUE top-k neighbors by exact Jaccard similarity (binary) using fast sparse math.
    inter = X_tr @ q^T
    union = nnz(tr) + nnz(q) - inter
    """
    inter = (X_tr_local @ q_row.T).toarray().ravel()
    nnz_tr = np.asarray(X_tr_local.getnnz(axis=1)).ravel()
    nnz_q = int(q_row.nnz)
    union = nnz_tr + nnz_q - inter

    sim = np.zeros_like(inter, dtype=float)
    mask = union > 0
    sim[mask] = inter[mask] / union[mask]

    kk = min(k, sim.size)
    idx = np.argpartition(-sim, kk - 1)[:kk]
    idx = idx[np.argsort(-sim[idx])]
    return idx


def topk_jaccard_indices_on_candidates(X_tr_local, q_row, cand_idx: np.ndarray, k: int) -> np.ndarray:
    """Exact Jaccard top-k, but only among candidate rows."""
    if cand_idx.size == 0:
        return np.array([], dtype=int)

    X_c = X_tr_local[cand_idx]
    inter = (X_c @ q_row.T).toarray().ravel()
    nnz_c = np.asarray(X_c.getnnz(axis=1)).ravel()
    nnz_q = int(q_row.nnz)
    union = nnz_c + nnz_q - inter

    sim = np.zeros_like(inter, dtype=float)
    mask = union > 0
    sim[mask] = inter[mask] / union[mask]

    kk = min(k, sim.size)
    idx_local = np.argpartition(-sim, kk - 1)[:kk]
    idx_local = idx_local[np.argsort(-sim[idx_local])]
    return cand_idx[idx_local]


def overlap_fraction(true_idx: np.ndarray, approx_idx: np.ndarray, k: int) -> float:
    """|TrueTopK ∩ ApproxTopK| / K"""
    if approx_idx.size == 0:
        return 0.0
    return len(set(true_idx.tolist()) & set(approx_idx.tolist())) / float(k)


# =========================
# Precompute MinHashes per num_perm and split
# =========================
def load_or_build_minhash_cache(X_mat, num_perm: int, split_tag: str):
    cache_path = f"minhash_cache_{split_tag}_numperm_{num_perm}.obj"
    if os.path.exists(cache_path):
        print(f"Loaded MinHash cache: {cache_path}")
        return joblib.load(cache_path)

    print(f"\n--- Precomputing MinHash for {split_tag} (num_perm={num_perm}) ---")
    t0 = time.perf_counter()
    all_mh = [None] * X_mat.shape[0]
    for i in tqdm(range(X_mat.shape[0]), desc=f"MinHash({split_tag}) perm={num_perm}", unit="doc"):
        all_mh[i] = minhash_from_sparse_row(X_mat[i], num_perm=num_perm)
    joblib.dump(all_mh, cache_path)
    print(f"Saved: {cache_path} | time={time.perf_counter()-t0:.2f}s")
    return all_mh


minhash_train = {}
minhash_test = {}
for p in NUM_PERM_VALUES:
    minhash_train[p] = load_or_build_minhash_cache(X_tr, p, f"train_{X_tr.shape[0]}")
    minhash_test[p] = load_or_build_minhash_cache(X_te, p, f"test_{X_te.shape[0]}")


# =========================
# Brute vs LSH (train/test split)
# =========================
rows = []

n_q = X_te.shape[0] if MAX_QUERIES is None else min(MAX_QUERIES, X_te.shape[0])
q_idx = np.arange(n_q, dtype=int)

print("\n================= Brute Force =================")
print("[Brute] Computing TRUE top-K and predictions...")
t_q = time.perf_counter()

true_topk = []
brute_preds = []

for i in tqdm(q_idx, desc="Brute queries", unit="q"):
    q_row = X_te[i]
    idx_true = topk_true_jaccard_indices(X_tr, q_row, K)
    true_topk.append(idx_true)
    brute_preds.append(majority_vote(y_tr[idx_true]))

brute_query_time = time.perf_counter() - t_q
brute_acc = accuracy_score(y_te[q_idx], brute_preds)

rows.append({
    "Split": "Train/Test",
    "Type": "Brute-Force-Jaccard",
    "BuildTime": 0.0,
    "QueryTime": brute_query_time,
    "TotalTime": brute_query_time,
    "fraction_trueK_found": 1.0,
    "Accuracy": brute_acc,
    "Parameters": f"K={K}"
})


# -------- LSH-Jaccard runs --------
for num_perm in NUM_PERM_VALUES:
    tr_mh = minhash_train[num_perm]
    te_mh = minhash_test[num_perm]

    for tau in TAU_VALUES:
        print(f"\n[LSH] num_perm={num_perm} tau={tau} | building index...")
        t_b = time.perf_counter()
        try:
            lsh = MinHashLSH(threshold=tau, num_perm=num_perm)
        except ValueError as e:
            print(f"[SKIP] LSH invalid params: num_perm={num_perm}, tau={tau} -> {e}")
            rows.append({
                "Split": "Train/Test",
                "Type": "LSH-Jaccard",
                "BuildTime": np.nan,
                "QueryTime": np.nan,
                "TotalTime": np.nan,
                "fraction_trueK_found": np.nan,
                "Accuracy": np.nan,
                "Parameters": f"K={K}, tau={tau}, num_perm={num_perm}, SKIPPED ({e})"
            })
            continue

        for local_i in tqdm(range(X_tr.shape[0]), desc="LSH insert", unit="doc"):
            lsh.insert(str(local_i), tr_mh[local_i])

        build_time = time.perf_counter() - t_b

        print("[LSH] querying + exact ranking on candidates...")
        t_q2 = time.perf_counter()

        lsh_preds = []
        fracs = []

        empty_cnt = 0
        cand_sizes = []

        majority_label = majority_vote(y_tr)

        for i in tqdm(q_idx, desc="LSH queries", unit="q"):
            q_row = X_te[i]
            q_mh = te_mh[i]

            cand_keys = lsh.query(q_mh)
            if not cand_keys:
                empty_cnt += 1
                lsh_preds.append(majority_label)
                fracs.append(0.0)
                continue

            cand_local = np.array([int(s) for s in cand_keys], dtype=int)
            cand_sizes.append(cand_local.size)

            approx_idx = topk_jaccard_indices_on_candidates(X_tr, q_row, cand_local, K)
            if approx_idx.size == 0:
                lsh_preds.append(majority_label)
                fracs.append(0.0)
                continue

            lsh_preds.append(majority_vote(y_tr[approx_idx]))
            fracs.append(overlap_fraction(true_topk[i], approx_idx, K))

        query_time = time.perf_counter() - t_q2
        total_time = build_time + query_time
        acc = accuracy_score(y_te[q_idx], lsh_preds)
        frac_mean = float(np.mean(fracs)) if fracs else 0.0

        avg_cands = float(np.mean(cand_sizes)) if cand_sizes else 0.0
        empty_rate = empty_cnt / float(n_q)

        rows.append({
            "Split": "Train/Test",
            "Type": "LSH-Jaccard",
            "BuildTime": build_time,
            "QueryTime": query_time,
            "TotalTime": total_time,
            "fraction_trueK_found": frac_mean,
            "Accuracy": acc,
            "Parameters": f"K={K}, tau={tau}, num_perm={num_perm}, empty_rate={empty_rate:.2%}, avg_cands={avg_cands:.1f}"
        })


# =========================
# Report tables
# =========================
df = pd.DataFrame(rows)

print("\n================= Results =================")
print(df[["Split", "Type", "BuildTime", "QueryTime", "TotalTime", "fraction_trueK_found", "Accuracy", "Parameters"]]
      .to_string(index=False))

# Save to CSV for your report
df.to_csv("NEWknn_lsh_train_test.csv", index=False)
print("\nSaved: NEWknn_lsh_train_test.csv")
