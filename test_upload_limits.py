from config import MAX_FILE_SIZE_BYTES, MAX_ROWS
from src.data.validator import (
    validate_file_size,
    validate_row_count
)

import pandas as pd


# -----------------------------
# Test 1: Valid file size
# -----------------------------

result = validate_file_size(
    10 * 1024 * 1024,  # 10 MB
    MAX_FILE_SIZE_BYTES
)

print("Valid file size:")
print(result)


# -----------------------------
# Test 2: Invalid file size
# -----------------------------

result = validate_file_size(
    60 * 1024 * 1024,  # 60 MB
    MAX_FILE_SIZE_BYTES
)

print("\nInvalid file size:")
print(result)


# -----------------------------
# Test 3: Valid row count
# -----------------------------

df_valid = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003"]
})

result = validate_row_count(
    df_valid,
    MAX_ROWS
)

print("\nValid row count:")
print(result)


# -----------------------------
# Test 4: Invalid row count
# -----------------------------

df_invalid = pd.DataFrame({
    "customer_id": range(MAX_ROWS + 1)
})

result = validate_row_count(
    df_invalid,
    MAX_ROWS
)

print("\nInvalid row count:")
print(result)