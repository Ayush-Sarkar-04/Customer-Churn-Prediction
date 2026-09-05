import pandas as pd

from src.data.validator import validate_data_types


# Correct transaction data
df = pd.DataFrame({
    "bill_amount": [500.00, "invalid", 750.25]
})

result = validate_data_types(
    df,
    "transaction"
)

print(result)