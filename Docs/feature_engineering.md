# Feature Engineering

This step prepared the fraud dataset for machine learning.

## Dataset columns used
- amount
- transaction_hour
- merchant_category
- foreign_transaction
- location_mismatch
- device_trust_score
- velocity_last_24h
- cardholder_age

## Target column
- is_fraud

## Steps performed
1. Loaded cleaned dataset
2. Removed transaction_id because it is only an identifier
3. Separated features and target
4. Detected categorical column: merchant_category
5. Applied one-hot encoding to convert text categories into numeric features
6. Split data into training and testing sets
7. Applied StandardScaler to normalize feature values
8. Saved processed training and testing datasets

## Why encoding was needed
Machine learning models and scalers require numeric values.
The merchant_category column contained text values such as Food, so it had to be converted into numeric columns.