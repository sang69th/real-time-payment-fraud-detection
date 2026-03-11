# Data Cleaning

This step cleaned the credit card fraud dataset before model training.

Checks performed:
- missing values check
- duplicate row check
- data type inspection
- fraud class distribution check

Cleaning actions:
- duplicate rows removed
- cleaned dataset saved to:
  `data/processed/creditcard_cleaned.csv`

Why this step matters:
- machine learning models need consistent and reliable input
- duplicate rows can bias learning
- checking data types helps ensure features are usable