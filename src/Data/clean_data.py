import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print("Dataset loaded successfully.")
print(df.head())

print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()
print("\nDataset shape after removing duplicates:")
print(df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nFraud class distribution:")
print(df["is_fraud"].value_counts())

fraud_ratio = df["is_fraud"].mean() * 100
print(f"\nFraud percentage after cleaning: {fraud_ratio:.4f}%")

df.to_csv("data/processed/creditcard_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")