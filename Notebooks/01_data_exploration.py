import pandas as pd
df = pd.read_csv("data/creditcard.csv")


print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


print("\nColumn information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFraud vs Non-Fraud count:")
print(df["is_fraud"].value_counts())

fraud_percentage = df["is_fraud"].mean() * 100
print(f"\nFraud percentage: {fraud_percentage:.4f}%")



import matplotlib.pyplot as plt

class_counts = df["is_fraud"].value_counts()

class_counts.plot(kind="bar")

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.show()

print("\nAverage transaction amount:")
print(df["amount"].mean())

print("\nAverage fraud transaction amount:")
print(df[df["is_fraud"] == 1]["amount"].mean())

print("\nAverage normal transaction amount:")
print(df[df["is_fraud"] == 0]["amount"].mean())


plt.hist(df["amount"], bins=50)

plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()


fraud_amounts = df[df["is_fraud"] == 1]["amount"]
normal_amounts = df[df["is_fraud"] == 0]["amount"]

plt.hist(normal_amounts, bins=50, alpha=0.5, label="Normal")
plt.hist(fraud_amounts, bins=50, alpha=0.5, label="Fraud")

plt.legend()
plt.title("Fraud vs Normal Transaction Amounts")
plt.xlabel("Amount")
plt.ylabel("Frequency")

plt.show()

plt.hist(df["transaction_hour"], bins=50)

plt.title("Transaction Time Distribution")
plt.xlabel("Time")
plt.ylabel("Number of Transactions")

plt.show()