import pandas as pd

from src.data.validator import validate_uploaded_datasets

customers_df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003"],
    "gender": ["Male", "Female", "Male"],
    "age": [25, 30, 28],
    "city": ["Delhi", "Mumbai", "Kolkata"],
    "registration_date": [
        "2026-01-10",
        "2026-02-15",
        "2026-03-20"
    ]
})

transactions_df = pd.DataFrame({
    "transaction_id": ["T001", "T002", "T003"],
    "customer_id": ["C001", "C999", "C003"],
    "transaction_date": [
        "2026-04-01",
        "2026-04-02",
        "2026-04-03"
    ],
    "bill_amount": [500.00, 750.00, 1000.00],
    "outlet": ["Delhi", "Mumbai", "Kolkata"]
})

campaigns_df = pd.DataFrame({
    "campaign_id": ["CAM001", "CAM002", "CAM003"],
    "customer_id": ["C001", "C002", "C003"],
    "campaign_date": [
        "2026-04-01",
        "2026-04-02",
        "2026-04-03"
    ],
    "campaign_type": [
        "SMS",
        "Email",
        "SMS"
    ],
    "reward_type": [
        "Points",
        "Discount",
        "Points"
    ],
    "reward_value": [100, 10, 150],
    "campaign_cost": [5.0, 10.0, 5.0],
    "sent": [1, 1, 1],
    "delivered": [1, 1, 1],
    "clicked": [1, 0, 1],
    "redeemed": [1, 0, 1],
    "redemption_date": [
        "2026-04-02",
        None,
        "2026-04-04"
    ]
})

result = validate_uploaded_datasets(
    customers_df,
    transactions_df,
    campaigns_df
)


print("INVALID UPLOAD VALIDATION RESULT:")
print(result)