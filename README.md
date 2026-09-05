# CUSTOMER CHURN PREDICTION AND CAMPAIGN ANALYTICS SYSTEM

A machine learning and customer analytics system designed to analyze customer behavior, identify churn patterns, segment customers, predict churn probability, and evaluate marketing campaign performance.

---

## PROJECT OVERVIEW

The system combines customer profiles, transaction history, and campaign activity to generate customer-level insights and prepare data for machine learning.

Main components:

- Data validation
- Data preprocessing
- Feature engineering
- RFM analysis
- Customer segmentation
- Churn prediction
- Campaign analytics
- Data visualization
- Interactive dashboard

---

## PROJECT STRUCTURE

    Customer-Churn-Prediction/
    ├── app/
    ├── data/
    │   ├── training/
    │   └── uploads/
    ├── models/
    ├── notebooks/
    ├── outputs/
    ├── src/
    │   ├── analytics/
    │   ├── data/
    │   ├── features/
    │   ├── ml/
    │   ├── preprocessing/
    │   └── visualization/
    ├── tests/
    ├── .gitignore
    ├── config.py
    ├── requirements.txt
    └── README.md

---

## DATASETS

### Customers

Customer profile information:

    customer_id
    gender
    age
    city
    registration_date

### Transactions

Purchase information:

    transaction_id
    customer_id
    transaction_date
    bill_amount
    outlet

### Campaigns

Campaign activity and engagement:

    campaign_id
    customer_id
    campaign_date
    campaign_type
    reward_type
    reward_value
    campaign_cost
    sent
    delivered
    clicked
    redeemed
    redemption_date

### Customer Features

`customer_features.csv` is a derived customer-level dataset used for machine-learning preparation and reference. It is not a raw upload schema.

---

## DATA VALIDATION

Uploaded CSV files must follow predefined schemas.

Validation includes:

- Required and unexpected columns
- Data types
- Dates
- Missing values
- Duplicates
- IDs
- Customer references
- Cross-file consistency

Resource limits are also applied:

    Maximum file size: 50 MB
    Maximum rows: 500,000
    Maximum uploaded files: 4

User uploads are excluded from Git version control.

---

## FEATURE ENGINEERING

The system generates customer-level behavioral features using information available up to the observation date.

### RFM

- **Recency** — Days since last purchase
- **Frequency** — Number of purchases
- **Monetary** — Total spending
- **Average Bill**
- **Average Purchase Gap**

### Customer Features

- Customer tenure
- Campaigns sent
- Campaigns delivered
- Campaigns clicked
- Campaigns redeemed
- Delivery rate
- Click rate
- Redemption rate
- Previous redemptions

Future data is excluded from feature calculations to prevent data leakage.

---

## CUSTOMER SEGMENTATION

Planned RFM segments:

- Champions
- Loyal Customers
- Potential Loyalists
- Regular Customers
- At Risk
- Inactive
- Lost

---

## CHURN DEFINITION

A customer is considered churned when they make no purchase during the following 90 days.

    churn = 1 → No purchase in future 90 days
    churn = 0 → At least one purchase in future 90 days

---

## MACHINE LEARNING

Planned models:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation will include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Feature Importance / Coefficients
- Model Comparison

The final system will generate churn probabilities and risk categories.

---

## CAMPAIGN ANALYTICS

Campaign performance is analyzed using:

    Sent → Delivered → Clicked → Redeemed

Key metrics:

    Delivery Rate = Delivered / Sent
    Click Rate = Clicked / Delivered
    Redemption Rate = Redeemed / Delivered

Campaigns will be compared by campaign type, reward type, and customer segment.

---

## TESTING

The `tests/` directory contains tests and verification scripts covering:

- Data validation
- Upload validation
- Resource limits
- Preprocessing
- Customer tenure
- RFM features
- Campaign features
- Previous redemptions
- Customer feature generation
- Churn labeling

---

## TECHNOLOGY STACK

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Joblib
- Streamlit
- SQLite / SQL
- Jupyter
- Git
- GitHub

---

## CURRENT STATUS

### Completed

- Data validation
- Upload validation
- Resource limits
- Data preprocessing
- Customer tenure
- RFM feature engineering
- Campaign features
- Previous redemptions
- Customer feature generation
- Churn labeling
- Testing structure
- Git/GitHub setup

### Planned

- ML dataset preparation
- Churn prediction models
- Model evaluation
- Risk categorization
- Customer segmentation
- Campaign analytics
- Visualizations
- Streamlit dashboard
- Full system integration

---

## INSTALLATION

Clone the repository:

    git clone https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction.git

    cd Customer-Churn-Prediction

Create a virtual environment:

    python -m venv .venv

Activate it on Windows:

    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

---

## DEVELOPMENT

The project is developed incrementally, with each major component tested before integration.

Git workflow:

    git status
    git add .
    git commit -m "Describe the change"
    git push

---

## REPOSITORY

GitHub:

https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction