import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Remove unnecessary column
df.drop("customerID", axis=1, inplace=True)

# Convert categorical columns
df = pd.get_dummies(df)

print(df.head())

print("\nFinal Shape:")
print(df.shape)
