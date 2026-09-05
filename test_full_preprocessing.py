import pandas as pd

from src.preprocessing.preprocessing import (
    clean_column_names,
    convert_dates,
    convert_numeric_columns,
    check_missing_values,
    remove_empty_rows
)


# Create sample data containing common data-quality issues
df = pd.DataFrame({
    " Customer_ID ": ["C001", "C002", None, "C004"],
    " BILL_AMOUNT ": ["500", "750.50", "invalid", "1000"],
    " Transaction_Date ": [
        "2026-01-01",
        "2026-01-02",
        "invalid-date",
        "2026-01-04"
    ],
    "Outlet ": ["Delhi", "Mumbai", None, "Kolkata"]
})


print("===== ORIGINAL DATA =====")
print(df)
print("\nOriginal columns:")
print(df.columns.tolist())


# 1. Clean column names
df = clean_column_names(df)

print("\n===== AFTER COLUMN CLEANING =====")
print(df.columns.tolist())


# 2. Convert dates
df = convert_dates(
    df,
    ["transaction_date"]
)

print("\n===== AFTER DATE CONVERSION =====")
print(df["transaction_date"])
print("Data type:", df["transaction_date"].dtype)


# 3. Convert numeric columns
df = convert_numeric_columns(
    df,
    ["bill_amount"]
)

print("\n===== AFTER NUMERIC CONVERSION =====")
print(df["bill_amount"])
print("Data type:", df["bill_amount"].dtype)


# 4. Check missing values
missing = check_missing_values(df)

print("\n===== MISSING VALUES =====")
print(missing)


# 5. Remove completely empty rows
df = remove_empty_rows(df)

print("\n===== FINAL CLEANED DATA =====")
print(df)

print("\nFinal row count:", len(df))