#!/usr/bin/env python3
import argparse
import time
import numpy as np
import joblib
import os
import csv
import math
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -----------------------
# Helpers (core logic)
# -----------------------

def choose_br_from_tau(num_perm: int, tau: float, min_b: int = 2):
    """
    Pick (b,r) with b*r=num_perm and b>=min_b so that the LSH 50% collision point s*
    is closest to tau.
    """
    best = None
    best_err = float("inf")

    for r in range(1, num_perm + 1):
        if num_perm % r != 0:
            continue
        b = num_perm // r
        if b < min_b:
            continue

        # s* where P(candidate)=0.5
        # s* = (1 - 0.5^(1/b))^(1/r)
        s_star = (1.0 - (0.5 ** (1.0 / b))) ** (1.0 / r)
        err = abs(s_star - tau)

        if err < best_err:
            best_err = err
            best = (b, r, s_star)

    if best is None:
        raise ValueError(f"No valid (b,r) found for num_perm={num_perm} with b>={min_b}")

    b, r, s_star = best
    return b, r, s_star

def minhash_from_sparse_row(row_csr: csr_matrix, num_perm: int) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for j in row_csr.indices:
        mh.update(str(j).encode("utf8"))
    return mh


def topk_bruteforce_jaccard_indices(X_train: csr_matrix, train_nnz: np.ndarray, q: csr_matrix, k: int) -> np.ndarray:
    q_nnz = q.nnz
    inter = X_train.multiply(q).sum(axis=1).A1
    union = train_nnz + q_nnz - inter

    sims = np.zeros_like(inter, dtype=np.float64)
    mask = union > 0
    sims[mask] = inter[mask] / union[mask]

    if k >= sims.size:
        return np.argsort(-sims)

    topk_unsorted = np.argpartition(-sims, kth=k - 1)[:k]
    return topk_unsorted[np.argsort(-sims[topk_unsorted])]


def topk_lsh_jaccard_indices(
    X_train: csr_matrix,
    train_nnz: np.ndarray,
    q: csr_matrix,
    lsh: MinHashLSH,
    num_perm: int,
    k: int
) -> np.ndarray:
    q_mh = minhash_from_sparse_row(q, num_perm=num_perm)
    cand_keys = lsh.query(q_mh)
    if not cand_keys:
        return np.array([], dtype=np.int64)

    cand_idx = np.fromiter((int(c) for c in cand_keys), dtype=np.int64)
    q_nnz = q.nnz

    X_cand = X_train[cand_idx]
    inter = X_cand.multiply(q).sum(axis=1).A1
    union = train_nnz[cand_idx] + q_nnz - inter

    sims = np.zeros_like(inter, dtype=np.float64)
    mask = union > 0
    sims[mask] = inter[mask] / union[mask]

    if cand_idx.size <= k:
        return cand_idx[np.argsort(-sims)]

    topk_unsorted = np.argpartition(-sims, kth=k - 1)[:k]
    return cand_idx[topk_unsorted[np.argsort(-sims[topk_unsorted])]]


def detect_text_column(df):
    if "Content" in df.columns:
        return "Content"
    if "text" in df.columns:
        return "text"
    raise ValueError("Could not find a text column. Expected 'Content' (or 'text').")


def default_lsh_params(num_perm: int):
    if num_perm == 16:
        return (4, 4)
    if num_perm == 32:
        return (8, 4)
    if num_perm == 64:
        return (16, 4)
    raise ValueError("Supported num_perm are 16, 32, 64 only in this script.")


def wrap_iter(it, desc: str, total: int = None):
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, leave=False)


# -----------------------
# Multiprocessing brute-force
# -----------------------
# NOTE: processes, not threads (GIL)
from multiprocessing import Pool

_G_X_train = None
_G_train_nnz = None
_G_X_test = None
_G_k = None

def _init_worker(X_train, train_nnz, X_test, k):
    global _G_X_train, _G_train_nnz, _G_X_test, _G_k
    _G_X_train = X_train
    _G_train_nnz = train_nnz
    _G_X_test = X_test
    _G_k = k

def _brute_chunk(arg):
    start, end = arg
    out = []
    for i in range(start, end):
        out.append(topk_bruteforce_jaccard_indices(_G_X_train, _G_train_nnz, _G_X_test[i], _G_k))
    return out

def parallel_brute_true_topks(X_train, train_nnz, X_test, k, n_jobs: int, chunk_size: int):
    n_test = X_test.shape[0]
    chunks = [(i, min(i + chunk_size, n_test)) for i in range(0, n_test, chunk_size)]

    with Pool(processes=n_jobs, initializer=_init_worker, initargs=(X_train, train_nnz, X_test, k)) as pool:
        if tqdm is None:
            parts = pool.map(_brute_chunk, chunks)
        else:
            parts = list(tqdm(pool.imap(_brute_chunk, chunks), total=len(chunks), desc="brute queries (mp)"))

    # flatten
    return [x for part in parts for x in part]


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Part 2: LSH speed-up for KNN (Jaccard) with parallel brute baseline.")
    parser.add_argument("train_obj", help="Processed training dataset .obj (joblib).")
    parser.add_argument("test_obj", help="Processed test dataset .obj (joblib).")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--tau", type=float, default=0.9, help="Label only (LSH uses bands/rows).")
    parser.add_argument("--min_df", type=int, default=5)
    parser.add_argument("--max_df", type=float, default=0.8)
    parser.add_argument("--max_features", type=int, default=1000)
    parser.add_argument("--ngrams", type=str, default="1,2")

    parser.add_argument("--out_csv", default="lsh_report.csv", help="CSV output for report table.")
    parser.add_argument("--cache_brute", default="brute_true_topks.joblib", help="Cache file for brute top-K results.")
    parser.add_argument("--force_brute", action="store_true", help="Recompute brute baseline even if cache exists.")

    parser.add_argument("--jobs", type=int, default=12, help="Number of worker processes for brute baseline.")
    parser.add_argument("--chunk_size", type=int, default=300, help="Test docs per task chunk (tune 200-1000).")
    args = parser.parse_args()

    # ---- Load ----
    train_df = joblib.load(args.train_obj)
    test_df = joblib.load(args.test_obj)
    text_col = detect_text_column(train_df)

    # ---- Vectorize ----
    nmin, nmax = (int(x.strip()) for x in args.ngrams.split(","))
    vectorizer = CountVectorizer(
        binary=True,
        ngram_range=(nmin, nmax),
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )
    X_train = vectorizer.fit_transform(train_df[text_col].fillna("").astype(str)).tocsr()
    X_test = vectorizer.transform(test_df[text_col].fillna("").astype(str)).tocsr()

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    train_nnz = np.asarray(X_train.getnnz(axis=1)).astype(np.float64)

    print(f"Train docs: {n_train} | Test docs: {n_test} | Vocab: {X_train.shape[1]}")
    print(f"K={args.k} | tau(label)={args.tau} | num_perm in [16,32,64] | brute jobs={args.jobs} chunk={args.chunk_size}")
    if tqdm is None:
        print("(tip: pip install tqdm for progress bars)")
    print()

    # =========================================================
    # 1) BRUTE FORCE BASELINE (PARALLEL + CACHED)
    # =========================================================
    if (os.path.exists(args.cache_brute) and not args.force_brute):
        print(f"Loading brute-force cache: {args.cache_brute}")
        brute_query_s, true_topks, cache_meta = joblib.load(args.cache_brute)

        # Basic sanity check
        if cache_meta.get("k") != args.k or cache_meta.get("n_test") != n_test or cache_meta.get("n_train") != n_train:
            print("Cache metadata mismatch (k/n_test/n_train). Recomputing brute baseline...")
            args.force_brute = True
        else:
            print(f"Loaded brute baseline time: {brute_query_s:.2f}s")

    if args.force_brute or not (os.path.exists(args.cache_brute)):
        print("Computing brute-force true top-K for ALL test docs (parallel).")
        t0 = time.perf_counter()
        true_topks = parallel_brute_true_topks(
            X_train, train_nnz, X_test, args.k,
            n_jobs=args.jobs,
            chunk_size=args.chunk_size
        )
        brute_query_s = time.perf_counter() - t0

        cache_meta = {
            "k": args.k,
            "n_test": n_test,
            "n_train": n_train,
            "vocab": X_train.shape[1],
            "vectorizer": {
                "ngrams": (nmin, nmax),
                "min_df": args.min_df,
                "max_df": args.max_df,
                "max_features": args.max_features,
            }
        }
        joblib.dump((brute_query_s, true_topks, cache_meta), args.cache_brute)
        print(f"Brute baseline done: {brute_query_s:.2f}s")
        print(f"Cached to: {args.cache_brute}")

    rows = []
    rows.append({
        "Type": "Brute-Force-Jaccard",
        "BuildTime": 0.0,
        "QueryTime": brute_query_s,
        "TotalTime": brute_query_s,
        "Fraction": 1.0,
        "Parameters": "-"
    })

    # =========================================================
    # 2) LSH runs (single process)
    # =========================================================
    for num_perm in [16, 32, 64]:
        b, r = default_lsh_params(num_perm)
        #b, r, s_star = choose_br_from_tau(num_perm,args.tau,min_b=2)
        if b * r != num_perm or b < 2:
            raise ValueError(f"Bad (b,r)=({b},{r}) for num_perm={num_perm}")

        # Build LSH
        t0 = time.perf_counter()
        lsh = MinHashLSH(num_perm=num_perm, params=(b, r))
        for i in wrap_iter(range(n_train), desc=f"build LSH (perm={num_perm})", total=n_train):
            mh = minhash_from_sparse_row(X_train[i], num_perm=num_perm)
            lsh.insert(str(i), mh)
        build_s = time.perf_counter() - t0

        # Query + overlap
        t1 = time.perf_counter()
        overlaps = []
        for i in wrap_iter(range(n_test), desc=f"LSH queries (perm={num_perm})", total=n_test):
            q = X_test[i]
            true_topk = true_topks[i]
            lsh_topk = topk_lsh_jaccard_indices(X_train, train_nnz, q, lsh, num_perm=num_perm, k=args.k)

            if lsh_topk.size == 0:
                overlaps.append(0.0)
            else:
                inter = np.intersect1d(true_topk, lsh_topk).size
                overlaps.append(inter / float(args.k))

        lsh_query_s = time.perf_counter() - t1
        total_s = build_s + lsh_query_s
        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

        rows.append({
            "Type": "LSH-Jaccard",
            "BuildTime": build_s,
            "QueryTime": lsh_query_s,
            "TotalTime": total_s,
            "Fraction": mean_overlap,
            "Parameters": f"Perm={num_perm}, (b,r)=({b},{r}), tau(label)={args.tau}"
        })

    # =========================================================
    # 3) Export CSV (assignment table)
    # =========================================================
    print(f"\nWriting report CSV: {args.out_csv}")
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Type", "BuildTime", "QueryTime", "TotalTime", "Fraction", "Parameters"])
        for row in rows:
            w.writerow([
                row["Type"],
                f"{row['BuildTime']:.6f}",
                f"{row['QueryTime']:.6f}",
                f"{row['TotalTime']:.6f}",
                f"{row['Fraction']:.6f}",
                row["Parameters"]
            ])

    # Also print a readable table to stdout
    print("\nReport table:")
    hdr = f"{'Type':<18} {'BuildTime':>10} {'QueryTime':>10} {'TotalTime':>10} {'Fraction':>10}  Parameters"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(
            f"{row['Type']:<18} "
            f"{row['BuildTime']:>10.2f} "
            f"{row['QueryTime']:>10.2f} "
            f"{row['TotalTime']:>10.2f} "
            f"{(100.0*row['Fraction']):>9.2f}%  "
            f"{row['Parameters']}"
        )


if __name__ == "__main__":
    main()

