import pandas as pd

from src.preprocessing.preprocessing import convert_dates

df = pd.DataFrame({
    "registration_date": [
        "2026-01-10",
        "2026-02-15",
        "2026-03-20"
    ]
})

result = convert_dates(
    df,
    ["registration_date"]
)

print("Original data type:")
print(df["registration_date"].dtype)

print("\nConverted data type:")
print(result["registration_date"].dtype)

print("\nConverted dates:")
print(result["registration_date"])