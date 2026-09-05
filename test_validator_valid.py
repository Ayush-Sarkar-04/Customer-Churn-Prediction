import pandas as pd

from src.data.schema import TRANSACTION_COLUMNS
from src.data.validator import validate_columns


df = pd.read_csv(
    "data/training/transactions.csv"
)

result = validate_columns(
    df,
    TRANSACTION_COLUMNS
)

print(result)