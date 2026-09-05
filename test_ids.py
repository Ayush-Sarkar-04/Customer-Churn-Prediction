import pandas as pd

from src.data.validator import validate_ids


# Intentionally invalid customer IDs
df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        None
    ]
})

result = validate_ids(
    df,
    "customer_id"
)

print(result)