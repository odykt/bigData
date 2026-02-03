# eval.py
# Generate Kaggle submission: ONLY Id + Predicted
# Assumes:
# - data_train.obj exists (preprocessed train, includes Label)
# - data_test.obj exists (preprocessed test, NO Label)
# - vectorizer.obj exists (CountVectorizer fitted on preprocessed TRAIN text)
# - clfBaselineBest.obj exists (LinearSVC trained on TRAIN BoW features)

import pandas as pd
import joblib

# -----------------------
# Load preprocessed data
# -----------------------
dataTrain = joblib.load("data_train.obj")   # has Label
dataTest  = joblib.load("data_test.obj")    # no Label

# -----------------------
# Load fitted vectorizer + trained classifier
# -----------------------
bow_vectorizer = joblib.load("vectorizer.obj")        # MUST be fitted on train
clf = joblib.load("clfBaselineBest.obj")              # MUST be trained on train

# -----------------------
# Transform TEST and predict
# -----------------------
X_test = bow_vectorizer.transform(
    dataTest["Content"].fillna("").astype(str)
)

test_pred = clf.predict(X_test)

# -----------------------
# Build submission (ONLY Id, Predicted)
# -----------------------
submission = pd.DataFrame({
    "Id": dataTest["Id"],
    "Predicted": test_pred
})

submission.to_csv("testSet_categories.csv", index=False)

# -----------------------
# Sanity checks (optional but useful)
# -----------------------
print("Saved testSet_categories.csv")
print("Submission shape:", submission.shape)
print(submission.head())

zero_rows = (X_test.getnnz(axis=1) == 0).sum()
print("All-zero test rows:", zero_rows, "/", X_test.shape[0])

