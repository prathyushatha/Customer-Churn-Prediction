import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print(df.shape)

print(df.columns)

print(df.info())

print(df.isnull().sum())

df.drop("customerID", axis=1, inplace=True)

print(df.head())

print(df.dtypes)
