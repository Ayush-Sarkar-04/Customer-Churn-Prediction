import pandas as pd
def load_csv(path):
    return pd.read_csv(path)
def load_training_data(base_path):
    customers = load_csv(
        base_path / "customers.csv"
    )
    transactions = load_csv(
        base_path / "transactions.csv"
    )
    campaigns = load_csv(
        base_path / "campaigns.csv"
    )
    customer_features = load_csv(
        base_path / "customer_features.csv"
    )
    return (
        customers,
        transactions,
        campaigns,
        customer_features
    )