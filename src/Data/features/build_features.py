import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
df = pd.read_csv("data/processed/creditcard_cleaned.csv")

print("Dataset loaded successfully.")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

# Separate features and target
# We drop 'is_fraud' because that is the answer column we want to predict
# We also drop 'transaction_id' because it is just an ID, not useful for learning fraud patterns
X = df.drop(["is_fraud", "transaction_id"], axis=1)
y = df["is_fraud"]

print("\nFeature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)

print("\nColumn data types:")
print(X.dtypes)

print("\nCategorical (text) columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())

# Convert text columns into numeric columns
# merchant_category is text, so we encode it
X = pd.get_dummies(X, drop_first=True)

print("\nFeature dataset shape after encoding:", X.shape)
print("\nEncoded column names:")
print(X.columns.tolist())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Scale numeric features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling complete.")

# Save processed arrays
np.save("data/processed/X_train.npy", X_train_scaled)
np.save("data/processed/X_test.npy", X_test_scaled)
np.save("data/processed/y_train.npy", y_train.to_numpy())
np.save("data/processed/y_test.npy", y_test.to_numpy())

# Save encoded feature names too
pd.Series(X.columns).to_csv("data/processed/feature_names.csv", index=False)

print("\nProcessed datasets saved successfully.")
print("Saved files:")
print("- data/processed/X_train.npy")
print("- data/processed/X_test.npy")
print("- data/processed/y_train.npy")
print("- data/processed/y_test.npy")
print("- data/processed/feature_names.csv")