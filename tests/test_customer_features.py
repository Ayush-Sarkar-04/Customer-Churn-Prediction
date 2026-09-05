import pandas as pd

from src.features.feature_engineering import build_customer_features


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

customers_df = pd.read_csv(
    "data/training/customers.csv"
)

transactions_df = pd.read_csv(
    "data/training/transactions.csv"
)

campaigns_df = pd.read_csv(
    "data/training/campaigns.csv"
)


# --------------------------------------------------
# OBSERVATION DATE
# --------------------------------------------------

observation_date = "2026-02-10"


# --------------------------------------------------
# BUILD CUSTOMER FEATURES
# --------------------------------------------------

result = build_customer_features(
    customers_df,
    transactions_df,
    campaigns_df,
    observation_date
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\nCUSTOMER FEATURE TABLE")
print("=" * 80)

print("Shape:")
print(result.shape)

print("\nColumns:")
print(result.columns.tolist())

print("\nFirst 10 customers:")
print(result.head(10))


# --------------------------------------------------
# CHECK FOR MISSING VALUES
# --------------------------------------------------

print("\nMISSING VALUES")
print("=" * 80)

print(
    result.isna().sum()
)


# --------------------------------------------------
# CHECK CUSTOMER COUNT
# --------------------------------------------------

print("\nCUSTOMER COUNT CHECK")
print("=" * 80)

print(
    "Original customers:",
    len(customers_df)
)

print(
    "Feature table customers:",
    len(result)
)


# --------------------------------------------------
# CHECK DUPLICATE CUSTOMERS
# --------------------------------------------------

print("\nDUPLICATE CUSTOMER CHECK")
print("=" * 80)

print(
    "Duplicate customer IDs:",
    result["customer_id"].duplicated().sum()
)