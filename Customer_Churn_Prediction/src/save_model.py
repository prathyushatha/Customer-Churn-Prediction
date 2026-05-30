import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

df = pd.get_dummies(
    df,
    drop_first=True
)

X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

pickle.dump(
    model,
    open("models/model.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

pickle.dump(
    X.columns.tolist(),
    open("models/columns.pkl", "wb")
)

print("Model Saved Successfully")
