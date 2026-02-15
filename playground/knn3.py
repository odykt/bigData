import os
import time
import numpy as np
import pandas as pd
import joblib

from tqdm import tqdm
from datasketch import MinHash, MinHashLSH

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# =========================
# CONFIG
# =========================
DATA_PATH = "data_train.obj"

BOW_READY = False
BOW_PATH = "bow_knn_jaccard.obj"
VEC_PATH = "vectorizer_knn_jaccard.obj"

K = 7

# Part 2 params to sweep
TAU_VALUES = [0.9, 0.85, 0.8]          # tune as needed
NUM_PERM_VALUES = [16, 32, 64]         # required in prompt

# Optional speed switch for debugging:
MAX_QUERIES_PER_FOLD = None  # e.g. 2000, or None for all val docs


# =========================
# Load data
# =========================
dataTrain = joblib.load(DATA_PATH)
X_text = dataTrain["Content"].fillna("").astype(str)
y = dataTrain["Label"].values


# =========================
# Build / Load Binary BoW
# =========================
print("\n--- Building/Loading Binary BoW ---")
if not BOW_READY:
    bow_vectorizer = CountVectorizer(
        binary=True,          # IMPORTANT for Jaccard/set semantics
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=1000     # raise later if you want; brute gets heavier
    )
    X = bow_vectorizer.fit_transform(X_text).tocsr()
    joblib.dump(X, BOW_PATH)
    joblib.dump(bow_vectorizer, VEC_PATH)
else:
    bow_vectorizer = joblib.load(VEC_PATH)
    X = joblib.load(BOW_PATH).tocsr()

print("X:", X.shape, "nnz:", X.nnz)
print("y:", y.shape)


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


def topk_true_jaccard_indices(X_tr, q_row, k: int) -> np.ndarray:
    """
    TRUE top-k neighbors by exact Jaccard similarity (binary) using fast sparse math.
    inter = X_tr @ q^T
    union = nnz(tr) + nnz(q) - inter
    """
    inter = (X_tr @ q_row.T).toarray().ravel()
    nnz_tr = np.asarray(X_tr.getnnz(axis=1)).ravel()
    nnz_q = int(q_row.nnz)
    union = nnz_tr + nnz_q - inter

    sim = np.zeros_like(inter, dtype=float)
    mask = union > 0
    sim[mask] = inter[mask] / union[mask]

    kk = min(k, sim.size)
    idx = np.argpartition(-sim, kk - 1)[:kk]
    idx = idx[np.argsort(-sim[idx])]
    return idx


def topk_jaccard_indices_on_candidates(X_tr, q_row, cand_idx: np.ndarray, k: int) -> np.ndarray:
    """Exact Jaccard top-k, but only among candidate rows."""
    if cand_idx.size == 0:
        return np.array([], dtype=int)

    X_c = X_tr[cand_idx]
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
# Precompute MinHashes ONCE per num_perm and save
# =========================
def load_or_build_minhash_cache(num_perm: int):
    cache_path = f"minhash_cache_numperm_{num_perm}.obj"
    if os.path.exists(cache_path):
        print(f"Loaded MinHash cache: {cache_path}")
        return joblib.load(cache_path)

    print(f"\n--- Precomputing MinHash for ALL docs (num_perm={num_perm}) ---")
    t0 = time.perf_counter()
    all_mh = [None] * X.shape[0]
    for i in tqdm(range(X.shape[0]), desc=f"MinHash(all) perm={num_perm}", unit="doc"):
        all_mh[i] = minhash_from_sparse_row(X[i], num_perm=num_perm)
    joblib.dump(all_mh, cache_path)
    print(f"Saved: {cache_path} | time={time.perf_counter()-t0:.2f}s")
    return all_mh


minhash_caches = {p: load_or_build_minhash_cache(p) for p in NUM_PERM_VALUES}


# =========================
# 5-fold CV: Brute vs LSH, with accuracy + fraction(trueK) + timing
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
    print(f"\n================= Fold {fold} =================")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    n_q = X_va.shape[0] if MAX_QUERIES_PER_FOLD is None else min(MAX_QUERIES_PER_FOLD, X_va.shape[0])
    q_idx = np.arange(n_q, dtype=int)

    # -------- Brute-Force-Jaccard (also produces TRUE top-K for fraction metric) --------
    print("\n[Brute] Computing TRUE top-K and predictions...")
    t_q = time.perf_counter()

    true_topk = []
    brute_preds = []

    for i in tqdm(q_idx, desc=f"Fold {fold} brute queries", unit="q"):
        q_row = X_va[i]
        idx_true = topk_true_jaccard_indices(X_tr, q_row, K)
        true_topk.append(idx_true)
        brute_preds.append(majority_vote(y_tr[idx_true]))

    brute_query_time = time.perf_counter() - t_q
    brute_acc = accuracy_score(y_va[q_idx], brute_preds)

    rows.append({
        "Fold": fold,
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
        all_mh = minhash_caches[num_perm]

        for tau in TAU_VALUES:
            print(f"\n[LSH] num_perm={num_perm} tau={tau} | building index...")
            # Build LSH index on train split (keys are LOCAL indices 0..n_train-1)
            t_b = time.perf_counter()
            #lsh = MinHashLSH(threshold=tau, num_perm=num_perm)
            try:
                lsh = MinHashLSH(threshold=tau, num_perm=num_perm)
            except ValueError as e:
                print(f"[SKIP] LSH invalid params: num_perm={num_perm}, tau={tau} -> {e}")
                rows.append({
                    "Fold": fold,
                    "Type": "LSH-Jaccard",
                    "BuildTime": np.nan,
                    "QueryTime": np.nan,
                    "TotalTime": np.nan,
                    "fraction_trueK_found": np.nan,
                    "Accuracy": np.nan,
                    "Parameters": f"K={K}, tau={tau}, num_perm={num_perm}, SKIPPED ({e})"
                })
                continue

            for local_i, global_i in tqdm(
                enumerate(tr_idx),
                total=len(tr_idx),
                desc=f"Fold {fold} LSH insert",
                unit="doc"
            ):
                lsh.insert(str(local_i), all_mh[int(global_i)])

            build_time = time.perf_counter() - t_b

            # Query + exact rank on candidates
            print(f"[LSH] querying + exact ranking on candidates...")
            t_q2 = time.perf_counter()

            lsh_preds = []
            fracs = []

            empty_cnt = 0
            cand_sizes = []

            # Majority fallback label (fast and explicit)
            majority_label = majority_vote(y_tr)

            for i in tqdm(q_idx, desc=f"Fold {fold} LSH queries", unit="q"):
                global_q = int(va_idx[i])
                q_row = X_va[i]
                q_mh = all_mh[global_q]

                cand_keys = lsh.query(q_mh)
                if not cand_keys:
                    # no candidates -> fast fallback
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
            acc = accuracy_score(y_va[q_idx], lsh_preds)
            frac_mean = float(np.mean(fracs)) if fracs else 0.0

            # diagnostics (helps you prove it's not "always majority")
            avg_cands = float(np.mean(cand_sizes)) if cand_sizes else 0.0
            empty_rate = empty_cnt / float(n_q)

            rows.append({
                "Fold": fold,
                "Type": "LSH-Jaccard",
                "BuildTime": build_time,
                "QueryTime": query_time,
                "TotalTime": total_time,
                "fraction_trueK_found": frac_mean,
                "Accuracy": acc,
                "Parameters": f"K={K}, tau={tau}, num_perm={num_perm}, empty_rate={empty_rate:.2%}, avg_cands={avg_cands:.1f}"
            })

# =========================
# Report tables: per-fold + averaged
# =========================
df = pd.DataFrame(rows)

print("\n================= Per-fold results =================")
print(df[["Fold", "Type", "BuildTime", "QueryTime", "TotalTime", "fraction_trueK_found", "Accuracy", "Parameters"]]
      .to_string(index=False))

# Average over folds per method+params
grp_cols = ["Type", "Parameters"]
agg = df.groupby(grp_cols).agg(
    BuildTime_mean=("BuildTime", "mean"),
    QueryTime_mean=("QueryTime", "mean"),
    TotalTime_mean=("TotalTime", "mean"),
    fraction_trueK_mean=("fraction_trueK_found", "mean"),
    Accuracy_mean=("Accuracy", "mean"),
    Accuracy_std=("Accuracy", "std"),
).reset_index()

print("\n================= Averaged over folds =================")
print(agg.to_string(index=False))

# Save to CSV for your report
df.to_csv("knn_lsh_cv_per_fold.csv", index=False)
agg.to_csv("knn_lsh_cv_avg.csv", index=False)
print("\nSaved: knn_lsh_cv_per_fold.csv and knn_lsh_cv_avg.csv")

