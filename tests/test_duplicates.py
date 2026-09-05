import pandas as pd

from src.data.validator import validate_duplicates


# Valid customer data with dublicate IDs
df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        "C002"
    ]
})

result = validate_duplicates(
    df,
    "customer_id"
)

print(result)