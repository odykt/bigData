#!/usr/bin/env python3
import argparse
import time
import numpy as np
import joblib

from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


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
    topk_sorted = topk_unsorted[np.argsort(-sims[topk_unsorted])]
    return topk_sorted


def topk_lsh_jaccard_indices(
    X_train: csr_matrix,
    train_nnz: np.ndarray,
    q: csr_matrix,
    lsh: MinHashLSH,
    num_perm: int,
    k: int,
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
    topk_sorted = topk_unsorted[np.argsort(-sims[topk_unsorted])]
    return cand_idx[topk_sorted]


def detect_text_column(df):
    if "Content" in df.columns:
        return "Content"
    if "text" in df.columns:
        return "text"
    raise ValueError("Could not find a text column. Expected 'Content' (or 'text').")


def default_lsh_params(num_perm: int):
    # Reasonable defaults with b>=2 and b*r=num_perm
    if num_perm == 16:
        return (4, 4)
    if num_perm == 32:
        return (8, 4)
    if num_perm == 64:
        return (16, 4)
    # fallback: try r=4 and b=num_perm/4 if divisible, else r=2
    if num_perm % 4 == 0 and (num_perm // 4) >= 2:
        return (num_perm // 4, 4)
    if num_perm % 2 == 0 and (num_perm // 2) >= 2:
        return (num_perm // 2, 2)
    raise ValueError(f"Cannot choose (b,r) for num_perm={num_perm} with b>=2 and b*r=num_perm.")


def wrap_iter(it, desc: str, total: int = None):
    if tqdm is None:
        return it
    return tqdm(it, desc=desc, total=total, leave=False)


def main():
    parser = argparse.ArgumentParser(description="Part 2: LSH speed-up for KNN (Jaccard) train->test.")
    parser.add_argument("train_obj", help="Processed training dataset .obj (joblib).")
    parser.add_argument("test_obj", help="Processed test dataset .obj (joblib).")
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--tau", type=float, default=0.9, help="Reported threshold label (LSH uses bands/rows params).")
    parser.add_argument("--min_df", type=int, default=5)
    parser.add_argument("--max_df", type=float, default=0.8)
    parser.add_argument("--max_features", type=int, default=1000)
    parser.add_argument("--ngrams", type=str, default="1,2")
    parser.add_argument("--bands", type=int, default=None, help="Override bands b (must satisfy b*r=num_perm).")
    parser.add_argument("--rows", type=int, default=None, help="Override rows r (must satisfy b*r=num_perm).")
    args = parser.parse_args()

    train_df = joblib.load(args.train_obj)
    test_df = joblib.load(args.test_obj)

    text_col = detect_text_column(train_df)

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

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    train_nnz = np.asarray(X_train.getnnz(axis=1)).astype(np.float64)

    num_perms = [16, 32, 64]

    print(f"Train docs: {n_train} | Test docs: {n_test} | Vocab: {X_train.shape[1]}")
    print(f"K={args.k} | tau(label)={args.tau} | ngrams=({nmin},{nmax}) | min_df={args.min_df} | max_df={args.max_df} | max_features={args.max_features}")
    if tqdm is None:
        print("(tip: pip install tqdm for progress bars)")
    print()

    header = "num_perm | (b,r)     | build_s | brute_query_s | lsh_query_s | speedup(q) | mean_topK_overlap"
    print(header)
    print("-" * len(header))

    for num_perm in num_perms:
        # choose (b,r)
        if args.bands is not None or args.rows is not None:
            if args.bands is None or args.rows is None:
                raise ValueError("If overriding LSH params, provide BOTH --bands and --rows.")
            b, r = args.bands, args.rows
        else:
            b, r = default_lsh_params(num_perm)

        if b * r != num_perm:
            raise ValueError(f"Invalid LSH params: b*r must equal num_perm. Got b={b}, r={r}, num_perm={num_perm}.")
        if b < 2:
            raise ValueError(f"Invalid LSH params: b must be >= 2. Got b={b}.")

        # ---- Build LSH over train ----
        t0 = time.perf_counter()
        lsh = MinHashLSH(num_perm=num_perm, params=(b, r))

        for i in wrap_iter(range(n_train), desc=f"build LSH (p={num_perm})", total=n_train):
            mh = minhash_from_sparse_row(X_train[i], num_perm=num_perm)
            lsh.insert(str(i), mh)

        build_s = time.perf_counter() - t0

        # ---- Brute force top-K for all test (baseline) ----
        t1 = time.perf_counter()
        true_topks = []
        for i in wrap_iter(range(n_test), desc=f"brute queries (p={num_perm})", total=n_test):
            q = X_test[i]
            true_topks.append(topk_bruteforce_jaccard_indices(X_train, train_nnz, q, args.k))
        brute_query_s = time.perf_counter() - t1

        # ---- LSH queries + overlap ----
        t2 = time.perf_counter()
        overlaps = []
        for i in wrap_iter(range(n_test), desc=f"LSH queries (p={num_perm})", total=n_test):
            q = X_test[i]
            true_topk = true_topks[i]
            lsh_topk = topk_lsh_jaccard_indices(X_train, train_nnz, q, lsh, num_perm=num_perm, k=args.k)

            if lsh_topk.size == 0:
                overlaps.append(0.0)
            else:
                inter = np.intersect1d(true_topk, lsh_topk).size
                overlaps.append(inter / float(args.k))

        lsh_query_s = time.perf_counter() - t2

        speedup_q = (brute_query_s / lsh_query_s) if lsh_query_s > 0 else float("inf")
        mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0

        print(
            f"{num_perm:7d} | "
            f"({b:2d},{r:2d})   | "
            f"{build_s:6.2f} | "
            f"{brute_query_s:12.2f} | "
            f"{lsh_query_s:10.2f} | "
            f"{speedup_q:9.2f} | "
            f"{mean_overlap:16.4f}"
        )


if __name__ == "__main__":
    main()

