# eval_prepare.py
# Preprocess the TEST set (no labels) using the SAME pipeline as training.
# Output: data_test.obj (joblib)

import nltk
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

from misc import load_and_process_data, clean_text

TEST_PATH = "../bigdata2025classification/test_without_labels.csv"
OUT_PATH = "data_test.obj"


def main() -> None:
    # Download required NLTK data if not already present
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Load test data (must match how you load train: delimiter/columns etc.)
    dataTest = load_and_process_data(TEST_PATH)

    # IMPORTANT: Do NOT drop duplicates in test (Kaggle expects one prediction per Id)
    for col in ["Title", "Content"]:
        if col not in dataTest.columns:
            raise KeyError(f"Missing expected column '{col}' in test set. Found: {list(dataTest.columns)}")

        dataTest[col] = (
            dataTest[col]
            .fillna("")
            .astype(str)
            .apply(clean_text, args=(stop_words, stemmer, lemmatizer,2))
        )

    # Optional: basic stats (handy for debugging / parity with train)
    dataTest["text_length"] = dataTest["Content"].apply(len)
    dataTest["word_count"] = dataTest["Content"].apply(lambda x: len(str(x).split()))
    dataTest["sentence_count"] = dataTest["Content"].apply(lambda x: len(str(x).split(".")))
    dataTest["avg_word_length"] = dataTest["Content"].apply(
        lambda x: float(np.mean([len(w) for w in str(x).split()])) if str(x).split() else 0.0
    )

    print("Prepared TEST set:", dataTest.shape)
    print("Columns:", list(dataTest.columns))

    joblib.dump(dataTest, OUT_PATH)
    print(f"Saved preprocessed test set to: {OUT_PATH}")


if __name__ == "__main__":
    main()

