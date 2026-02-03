import pandas as pd
import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
from misc import *
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# -----------------------
# Load training data
# -----------------------
dataTrain = joblib.load("data_train.obj")

print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())

# IMPORTANT:
# Kaggle-style "test.csv" typically has NO labels.
# So we evaluate using 5-fold CV on the TRAIN set.


# -----------------------
# Load or build Bow
# -----------------------
print("medium pruning, min df=5, max df = .8 max features=200k, ngram 1,2, SVC-C=0.1")
bowReady = False
if not bowReady:
    bow_vectorizer = CountVectorizer(ngram_range=(1,2),min_df=5, max_df=0.8,max_features=200_000,binary=True)  # speed/quality win
    dataTrain_bow = bow_vectorizer.fit_transform(dataTrain["Content"].fillna("").astype(str))
    joblib.dump(dataTrain_bow, "bow.obj")
    joblib.dump(bow_vectorizer, "vectorizer.obj")
else:
    bow_vectorizer = joblib.load("vectorizer.obj")
    dataTrain_bow = joblib.load("bow.obj")

X = dataTrain_bow
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

    clf = LinearSVC(C=0.3, max_iter=10_000)  #
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


# dataTest = pd.read_csv("../bigdata2025classification/test.csv", sep="|")  # use sep="|" if needed
# X_test = bow_vectorizer.transform(dataTest["Content"].fillna("").astype(str))
# test_pred = LinearSVC().fit(X, y).predict(X_test)  # fit once on full train, then predict
# dataTest["prediction"] = test_pred
# dataTest.to_csv("test_with_predictions.csv", index=False)
#


