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
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.8,
        max_features=200_000,
        sublinear_tf=True,   # usually helps
        use_idf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(dataTrain["Content"].fillna("").astype(str))

    joblib.dump(X_tfidf, "tfidf.obj")
    joblib.dump(tfidf_vectorizer, "vectorizer.obj")
else:
    tfidf_vectorizer = joblib.load("vectorizer.obj")
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

    clf = LinearSVC(C=10, max_iter=10_000, class_weight='balanced')
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_va)
    acc = accuracy_score(y_va, pred)
    fold_acc.append(acc)

    all_true.extend(y_va)
    all_pred.extend(pred)

    print(f"Fold {fold} accuracy: {acc:.4f}")

joblib.dump(clf,"clfBaselineBest.obj")
print("\nMean CV accuracy:", np.mean(fold_acc), "+/-", np.std(fold_acc))
print("\nOverall classification report (aggregated over folds):")
print(classification_report(all_true, all_pred))
print("Confusion matrix (aggregated over folds):")
print(confusion_matrix(all_true, all_pred))


#--- Shapes ---
#X: (111220, 200000)
#y: (111220,)
#Fold 1 accuracy: 0.9695
#Fold 2 accuracy: 0.9711
#Fold 3 accuracy: 0.9693
#Fold 4 accuracy: 0.9706
#Fold 5 accuracy: 0.9724
#
#Mean CV accuracy: 0.9705718395971947 +/- 0.0011145447838619847
#
#Overall classification report (aggregated over folds):
#               precision    recall  f1-score   support
#
#     Business       0.95      0.95      0.95     24742
#Entertainment       0.99      0.99      0.99     44527
#       Health       0.98      0.97      0.98     11953
#   Technology       0.96      0.96      0.96     29998
#
#     accuracy                           0.97    111220
#    macro avg       0.97      0.97      0.97    111220
# weighted avg       0.97      0.97      0.97    111220
#
#Confusion matrix (aggregated over folds):
#[[23410   190   131  1011]
# [  224 44081    69   153]
# [  185   120 11595    53]
# [  921   188    28 28861]]
#



#############
### ONLY WITH STOP WORDS REMOVAL // NO STEMMING OR LEMMATIZATION
### slightly better
###########
#--- Shapes ---
#X: (111220, 200000)
#y: (111220,)
#Fold 1 accuracy: 0.9711
#Fold 2 accuracy: 0.9726
#Fold 3 accuracy: 0.9722
#Fold 4 accuracy: 0.9713
#Fold 5 accuracy: 0.9724
#
#Mean CV accuracy: 0.9719295090811005 +/- 0.0006121936533195175
#
#Overall classification report (aggregated over folds):
#               precision    recall  f1-score   support
#
#     Business       0.95      0.95      0.95     24742
#Entertainment       0.99      0.99      0.99     44527
#       Health       0.98      0.97      0.98     11953
#   Technology       0.96      0.96      0.96     29998
#
#     accuracy                           0.97    111220
#    macro avg       0.97      0.97      0.97    111220
# weighted avg       0.97      0.97      0.97    111220
#
#Confusion matrix (aggregated over folds):
#[[23467   153   140   982]
# [  222 44079    68   158]
# [  188    94 11624    47]
# [  862   177    31 28928]]

#####
##### Lemma + StopWordsRemoval (not stemming)
#y: (111220,)
#Fold 1 accuracy: 0.9706
#Fold 2 accuracy: 0.9730
#Fold 3 accuracy: 0.9718
#Fold 4 accuracy: 0.9714
#Fold 5 accuracy: 0.9730
#
#Mean CV accuracy: 0.9719654738356411 +/- 0.0009568753962876256
#
#Overall classification report (aggregated over folds):
#               precision    recall  f1-score   support
#
#     Business       0.95      0.95      0.95     24742
#Entertainment       0.99      0.99      0.99     44527
#       Health       0.98      0.97      0.98     11953
#   Technology       0.96      0.96      0.96     29998
#
#     accuracy                           0.97    111220
#    macro avg       0.97      0.97      0.97    111220
# weighted avg       0.97      0.97      0.97    111220
#
#Confusion matrix (aggregated over folds):
#[[23474   164   135   969]
# [  225 44083    65   154]
# [  193    91 11623    46]
# [  861   181    34 28922]]
#
#
