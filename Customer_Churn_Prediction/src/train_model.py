import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ==========================
# REMOVE UNNECESSARY COLUMN
# ==========================

df.drop("customerID", axis=1, inplace=True)

# ==========================
# HANDLE TotalCharges COLUMN
# ==========================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# ==========================
# CONVERT CATEGORICAL DATA
# ==========================

df = pd.get_dummies(
    df,
    drop_first=True
)

# ==========================
# FEATURES AND TARGET
# ==========================

X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# FEATURE SCALING
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================
# LOGISTIC REGRESSION MODEL
# ==========================

model = LogisticRegression(
    max_iter=1000
)

# ==========================
# TRAIN MODEL
# ==========================

model.fit(
    X_train,
    y_train
)

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# MODEL EVALUATION
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(accuracy)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(confusion_matrix(
    y_test,
    y_pred
))

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(
    y_test,
    y_pred
))
