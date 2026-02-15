import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# -----------------------
# Load training data
# -----------------------
dataTrain = joblib.load("data_train.obj")

print("\n--- Content Statistics ---")
print(dataTrain[['text_length', 'word_count', 'sentence_count', 'avg_word_length']].describe())


# -----------------------
# Load or build TF-IDF (SAME filenames)
# -----------------------
print("SCALING STRATEGY: TF-IDF + SGDClassifier (hinge loss)")

tfidfReady = False

if not tfidfReady:
    bow_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=200_000,
        sublinear_tf=True,
        use_idf=True
    )

    X = bow_vectorizer.fit_transform(
        dataTrain["Content"].fillna("").astype(str)
    )

    # KEEP SAME NAMES
    joblib.dump(X, "tfidf.obj")          # optional cache
    joblib.dump(bow_vectorizer, "vectorizer.obj")

else:
    bow_vectorizer = joblib.load("vectorizer.obj")
    X = joblib.load("tfidf.obj")

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

    # Linear SVM via SGD (scales much better)
    clf = SGDClassifier(
        loss="hinge",        # SVM
        penalty="l2",
        alpha=1e-4,          # analogous to 1/C (tune if needed)
        max_iter=2000,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

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


# -----------------------
# Train FINAL model on full data
# -----------------------
final_clf = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=1e-4,
    max_iter=2000,
    tol=1e-3,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

final_clf.fit(X, y)

# KEEP SAME NAME so eval.py works
joblib.dump(final_clf, "clfBaselineBest.obj")

print("\nSaved vectorizer.obj and clfBaselineBest.obj (eval.py compatible)")

