import pandas as pd

from src.data.validator import validate_dataset
from src.data.schema import TRANSACTION_COLUMNS


# Load the real training transaction dataset
df = pd.read_csv(
    "data/training/transactions.csv"
)


# Intentionally remove a required column
df = df.drop(columns=["outlet"])


# Run the complete dataset validator
result = validate_dataset(
    df,
    "transaction",
    TRANSACTION_COLUMNS,
    "transaction_id"
)


# Display the validation result
print(result)