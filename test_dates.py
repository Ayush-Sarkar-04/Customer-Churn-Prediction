import pandas as pd

from src.data.validator import validate_dates


# invalid transaction dates
df = pd.DataFrame({
    "transaction_date": [
    "2026-06-10",
    "2026-07-15",
    "not-a-date"
]
})

result = validate_dates(
    df,
    "transaction"
)

print(result)