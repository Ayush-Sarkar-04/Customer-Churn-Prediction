import pandas as pd

from src.data.schema import TRANSACTION_COLUMNS
from src.data.validator import validate_columns


# Create an intentionally incorrect CSV structure
df = pd.DataFrame(columns=[
    "transaction_id",
    "customer_id",
    "transaction_date",
    "bill_amount",
    "wrong_column"
])


result = validate_columns(
    df,
    TRANSACTION_COLUMNS
)

print(result)