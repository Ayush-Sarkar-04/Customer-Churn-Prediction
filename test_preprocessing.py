import pandas as pd

from src.preprocessing.preprocessing import clean_column_names


df = pd.DataFrame({
    " Customer_ID ": ["C001", "C002"],
    " BILL_AMOUNT ": [500, 750],
    " Transaction_Date ": ["2026-01-01", "2026-01-02"]
})


result = clean_column_names(df)

print("Original columns:")
print(df.columns.tolist())

print("\nCleaned columns:")
print(result.columns.tolist())