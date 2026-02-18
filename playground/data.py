#!/usr/bin/env python3
import argparse
import os
import sys

import nltk
import numpy as np
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from misc import (
    load_and_process_data,
    check_duplicates,
    check_title_duplicates,
    check_content_duplicates,
    check_title_content_duplicates,
    check_column_types,
    clean_text,
)

REQUIRED_NLTK = ["stopwords", "wordnet", "omw-1.4"]


def ensure_nltk():
    """Download required NLTK corpora if missing."""
    for pkg in REQUIRED_NLTK:
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg != "omw-1.4" else "corpora/omw-1.4")
        except LookupError:
            nltk.download(pkg)


def infer_dataset_type(df):
    """
    Heuristic:
      - Train: has Category (and typically Id)
      - Test: no Category (and may have no Id)
    """
    cols = set(df.columns)
    if "Category" in cols:
        return "train"
    return "test"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset: load, clean (stem/lemmatize), add stats, save as joblib obj."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV filename (train.csv or test.csv)."
    )
    parser.add_argument(
        "output_obj",
        help="Output filename for joblib dump (e.g., data_train.obj or data_test.obj)."
    )
    args = parser.parse_args()

    input_csv = args.input_csv
    output_obj = args.output_obj

    if not os.path.exists(input_csv):
        print(f"[ERROR] Input CSV not found: {input_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading and processing data from: {input_csv}")
    data = load_and_process_data(input_csv)

    ds_type = infer_dataset_type(data)
    print(f"[INFO] Detected dataset type: {ds_type.upper()}")

    # ==== Checks & duplicate removal (train) ====
    if ds_type == "train":
        # Equivalent to your original script
        check_duplicates(data)
        check_title_duplicates(data)
        check_content_duplicates(data)
        check_title_content_duplicates(data)
        check_column_types(data)

        # Remove duplicates based on Title+Content (keep first)
        data = data.drop_duplicates(subset=["Title", "Content"], keep="first")
        print("\nDuplicates based on Title and Content removed. Data shape:", data.shape)

        data = data.reset_index(drop=True)
        print("\nIndex reset. Data shape:", data.shape)
        data.info()

    else:
        # Equivalent messaging for test (no labels / ids possibly)
        print("[INFO] Test set: skipping label/duplicate diagnostic checks (not applicable).")
        print("[INFO] Running basic column type check only.")
        check_column_types(data)

        # Still safe to de-duplicate Title+Content if present (optional but reasonable)
        if {"Title", "Content"}.issubset(set(data.columns)):
            before = data.shape
            data = data.drop_duplicates(subset=["Title", "Content"], keep="first")
            print("\nDuplicates based on Title and Content removed. Data shape:", data.shape, f"(was {before})")

            data = data.reset_index(drop=True)
            print("\nIndex reset. Data shape:", data.shape)
            data.info()
        else:
            print("[WARN] Test set missing Title/Content columns; skipping de-duplication and stats.")
            data.info()

    # ==== NLTK setup ====
    print("[INFO] Ensuring required NLTK resources...")
    ensure_nltk()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # ==== Cleaning ====
    if {"Title", "Content"}.issubset(set(data.columns)):
        print("Stem and lemmatize\n")
        for col in ["Title", "Content"]:
            data[col] = data[col].astype(str).apply(
                clean_text, args=(stop_words, stemmer, lemmatizer, 2)
            )
    else:
        print("[WARN] Missing Title/Content; skipping stemming/lemmatization.")

    # ==== Statistics (same as your script; safe-guarded) ====
    if "Content" in data.columns:
        print("get statistics")

        data["text_length"] = data["Content"].apply(len)
        data["word_count"] = data["Content"].apply(lambda x: len(str(x).split()))
        data["sentence_count"] = data["Content"].apply(lambda x: len(str(x).split(".")))
        data["avg_word_length"] = data["Content"].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0.0
        )

        print("\n--- Content Statistics ---")
        print(data[["text_length", "word_count", "sentence_count", "avg_word_length"]].describe())
    else:
        print("[WARN] Content column not found; skipping statistics.")

    # ==== Save ====
    joblib.dump(data, output_obj)
    print(f"[INFO] Saved processed dataset to: {output_obj}")


if __name__ == "__main__":
    main()

