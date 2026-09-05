import pandas as pd

from src.preprocessing.preprocessing import check_missing_values

df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003"],
    "age": [25, None, 30],
    "city": ["Delhi", "Mumbai", None]
})

result = check_missing_values(df)

print("Missing values by column:")
print(result)