import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

print(df.head())

print(df.shape)

sns.countplot(x='Churn', data=df)

plt.title("Churn Distribution")

plt.show()

sns.countplot(x='gender', hue='Churn', data=df)

plt.title("Gender vs Churn")

plt.show()

sns.countplot(x='Contract', hue='Churn', data=df)

plt.title("Contract Type vs Churn")

plt.xticks(rotation=20)

plt.show()

sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

plt.title("Monthly Charges vs Churn")

plt.show()

sns.boxplot(x='Churn', y='tenure', data=df)

plt.title("Tenure vs Churn")

plt.show()

plt.hist(df['MonthlyCharges'])

plt.title("Monthly Charges Distribution")

plt.xlabel("Monthly Charges")

plt.ylabel("Frequency")

plt.show()
df = pd.get_dummies(df)

plt.figure(figsize=(20, 12))

sns.heatmap(df.corr())

plt.title("Correlation Heatmap")

plt.show()