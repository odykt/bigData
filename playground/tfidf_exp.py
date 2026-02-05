import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# -----------------------
# Load training data
# -----------------------
dataTrain = joblib.load("data_train.obj")

print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())


# -----------------------
# Load or build TF-IDF
# -----------------------
print("TF-IDF: min_df=5, max_df=0.8, max_features=200k, ngram=(1,2), LinearSVC C=0.1")

tfidfReady = False

if not tfidfReady:
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=200_000,
        sublinear_tf=True,   # usually helps
        use_idf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(dataTrain["Content"].fillna("").astype(str))

    joblib.dump(X_tfidf, "tfidf.obj")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.obj")
else:
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.obj")
    X_tfidf = joblib.load("tfidf.obj")

X = X_tfidf
y = dataTrain["Label"].values

print("\n--- Shapes ---")
print("X:", X.shape)
print("y:", y.shape)


# -----------------------
# 5-fold Stratified CV (accuracy + report)
# -----------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc = []
all_true = []
all_pred = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = LinearSVC(C=0.1, max_iter=10_000)
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_va)
    acc = accuracy_score(y_va, pred)
    fold_acc.append(acc)

    all_true.extend(y_va)
    all_pred.extend(pred)

    print(f"Fold {fold} accuracy: {acc:.4f}")

print("\nMean CV accuracy:", np.mean(fold_acc), "+/-", np.std(fold_acc))
print("\nOverall classification report (aggregated over folds):")
print(classification_report(all_true, all_pred))
print("Confusion matrix (aggregated over folds):")
print(confusion_matrix(all_true, all_pred))



