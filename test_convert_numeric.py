import pandas as pd

from src.preprocessing.preprocessing import convert_numeric_columns

df = pd.DataFrame({
    "bill_amount": [
        "500",
        "750.50",
        "1000"
    ]
})

result = convert_numeric_columns(
    df,
    ["bill_amount"]
)

print("Original data type:")
print(df["bill_amount"].dtype)

print("\nConverted data type:")
print(result["bill_amount"].dtype)

print("\nConverted values:")
print(result["bill_amount"])