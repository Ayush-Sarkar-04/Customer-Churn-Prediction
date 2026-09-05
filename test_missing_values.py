import pandas as pd

from src.data.validator import validate_missing_values


# Valid customer data
df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003"],
    "gender": ["Male", "Female", "Male"],
    "age": [25, 30, 28],
    "city": ["Delhi", "Mumbai", None],
    "registration_date": [
        "2026-01-10",
        "2026-02-15",
        "2026-03-20"
    ]
})

required_columns = [
    "customer_id",
    "gender",
    "age",
    "city",
    "registration_date"
]

result = validate_missing_values(
    df,
    required_columns
)

print(result)