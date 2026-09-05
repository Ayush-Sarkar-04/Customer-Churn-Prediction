import pandas as pd

from src.preprocessing.preprocessing import remove_empty_rows

df = pd.DataFrame({
    "customer_id": [
        "C001",
        None,
        "C003",
        None
    ],
    "age": [
        25,
        None,
        30,
        None
    ],
    "city": [
        "Delhi",
        None,
        "Mumbai",
        None
    ]
})

print("Original data:")
print(df)

print("\nOriginal row count:")
print(len(df))

result = remove_empty_rows(df)
print("\nCleaned data:")
print(result)

print("\nCleaned row count:")
print(len(result))