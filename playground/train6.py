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
    bow_vectorizer = CountVectorizer(ngram_range=(1,2),min_df=5, max_df=0.8,max_features=200_000)  # speed/quality win
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

    clf = LinearSVC(C=0.1, max_iter=10_000)  #
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

joblib.dump(clf,"clfBaselineBest.obj")

dataTest = pd.read_csv("../bigdata2025classification/test_without_labels.csv")  # use sep="|" if needed
X_val = bow_vectorizer.transform(dataTest["Content"].fillna("").astype(str))
y_val_pred = LinearSVC().fit(X, y).predict(X_val)  # fit once on full train, then predict
submission = pd.DataFrame({
    "Id": dataTest["Id"],
    "Predicted": y_val_pred
})

# Save exactly as required
submission.to_csv("testSet_categories.csv", index=False)
#dataTest.to_csv("test_with_predictions.csv", index=False)
#


#--- Shapes ---
#X: (111220, 200000)
#y: (111220,)
#Fold 1 accuracy: 0.9573
#Fold 2 accuracy: 0.9603
#Fold 3 accuracy: 0.9603
#Fold 4 accuracy: 0.9592
#Fold 5 accuracy: 0.9591
#
#Mean CV accuracy: 0.9592699154828267 +/- 0.0011037576723196601
#
#Overall classification report (aggregated over folds):
#               precision    recall  f1-score   support
#
#     Business       0.93      0.93      0.93     24742
#Entertainment       0.98      0.98      0.98     44527
#       Health       0.97      0.95      0.96     11953
#   Technology       0.95      0.95      0.95     29998
#
#     accuracy                           0.96    111220
#    macro avg       0.96      0.95      0.96    111220
# weighted avg       0.96      0.96      0.96    111220
#
#Confusion matrix (aggregated over folds):
#[[23007   297   189  1249]
# [  325 43854   100   248]
# [  272   184 11406    91]
# [ 1181   334    60 28423]]
##
