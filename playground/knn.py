import numpy as np
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# -----------------------
# Load training data (prepared with temperature=2)
# -----------------------
dataTrain = joblib.load("data_train.obj")

print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())

# -----------------------
# Build Binary BoW for Jaccard
# -----------------------
print("KNN+Jaccard: Binary BoW, min_df=5, max_df=0.8, max_features=50k, ngram=(1,1)")
bowReady = False

if not bowReady:
    bow_vectorizer = CountVectorizer(
        binary=True,            # IMPORTANT for Jaccard
        ngram_range=(1, 2),     # start with unigrams (much faster)
        min_df=5,
        max_df=0.8,
        max_features=1000     # keep small first, KNN is expensive
    )
    X_bow = bow_vectorizer.fit_transform(dataTrain["Content"].fillna("").astype(str))
    joblib.dump(X_bow, "bow_knn_jaccard.obj")
    joblib.dump(bow_vectorizer, "vectorizer_knn_jaccard.obj")
else:
    bow_vectorizer = joblib.load("vectorizer_knn_jaccard.obj")
    X_bow = joblib.load("bow_knn_jaccard.obj")

X = X_bow
y = dataTrain["Label"].values

print("\n--- Shapes ---")
print("X:", X.shape)
print("y:", y.shape)

# -----------------------
# 5-fold Stratified CV
# -----------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc = []
all_true = []
all_pred = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # IMPORTANT: brute force for sparse + jaccard
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric="jaccard",
        algorithm="brute",
        n_jobs=-1
    )
    X_tr=X_tr.toarray()
    #y_tr=y_tr.toarray()
    knn.fit(X_tr, y_tr)
    
    X_va=X_va.toarray()
    pred = knn.predict(X_va)
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


#max features 100
#Fold 5 accuracy: 0.6996
#
#Mean CV accuracy: 0.7024545944973927 +/- 0.002598842378699728
#
#Overall classification report (aggregated over folds):
#               precision    recall  f1-score   support
#
#     Business       0.62      0.73      0.67     24742
#Entertainment       0.76      0.88      0.81     44527
#       Health       0.68      0.55      0.61     11953
#   Technology       0.69      0.48      0.57     29998
#
#     accuracy                           0.70    111220
#    macro avg       0.69      0.66      0.67    111220
# weighted avg       0.70      0.70      0.69    111220
#
#Confusion matrix (aggregated over folds):
#[[18017  2750   945  3030]
# [ 2496 38971   702  2358]
# [ 2026  2228  6591  1108]
# [ 6456  7488  1506 14548]]
#
