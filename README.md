# Customer Churn Prediction & Campaign Analytics System

> An end-to-end customer intelligence and churn decision-support system for understanding customer behavior, predicting churn risk, prioritizing retention opportunities, and analyzing marketing campaign performance.

**Built with:** Python · Pandas · NumPy · scikit-learn · Plotly · Streamlit

---

## Overview

This project combines customer profiles, transaction history, campaign activity, machine-learning predictions, RFM segmentation, retention prioritization, and campaign analytics into an integrated analytical application.

The workflow runs from controlled CSV ingestion through validation, preprocessing, feature engineering, churn modeling, customer intelligence, campaign analysis, and interactive Streamlit decision support.

> **Project positioning:** This is a portfolio-oriented decision-support system. It is not an autonomous campaign execution platform or a production deployment.

---

## Key Capabilities

| Capability | What it does |
|---|---|
| **Data Quality** | Validates uploaded customer, transaction, and campaign data before analysis |
| **Feature Engineering** | Builds customer-level behavioral, transaction, and campaign features |
| **Churn Prediction** | Estimates the probability that a customer will churn |
| **Risk Classification** | Converts churn probability into business-facing risk levels |
| **RFM Segmentation** | Groups customers using recency, frequency, and monetary behavior |
| **Retention Priority** | Combines churn risk and customer value into a 0–100 priority score |
| **Expected Revenue at Risk** | Quantifies customer value exposed to predicted churn |
| **Campaign Analytics** | Measures the Sent → Delivered → Clicked → Redeemed funnel |
| **Campaign Affinity** | Analyzes observed customer response by campaign type |
| **Feature Importance** | Exposes Random Forest feature importance as descriptive model information |
| **Customer Search** | Provides customer-level exploration and current-profile lookup |
| **Interactive Dashboard** | Brings the analytical workflow together in 8 Streamlit views |

---

## System Architecture

```text
Customer CSVs
     │
     ▼
┌─────────────────────┐
│ Schema Validation   │
│ Data Quality Checks │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Preprocessing       │
│ Type / Date Cleanup │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Feature Engineering │
│ RFM + Behavior      │
│ Campaign Features   │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 90-Day Churn Label  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Machine Learning    │
│ LR / DT / RF        │
└──────────┬──────────┘
           ▼
┌──────────────────────────────┐
│ Customer Intelligence        │
│ Probability / Risk / RFM     │
│ Retention Priority / Value   │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Campaign Analytics           │
│ Funnel + Campaign Affinity   │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ Streamlit Decision Support   │
└──────────────────────────────┘
```

---

## Data & Validation

The system works with three primary raw CSV datasets.

### Customers

Customer profile information:

```text
customer_id
gender
age
city
registration_date
```

### Transactions

Purchase information:

```text
transaction_id
customer_id
transaction_date
bill_amount
outlet
```

### Campaigns

Campaign delivery and engagement information:

```text
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
```

### Derived Customer Features

`customer_features.csv` is a derived customer-level dataset used for machine-learning preparation and reference. It is not a raw upload schema.

### Validation

Uploaded files are checked for:

- Required and unexpected columns
- Data types
- Date validity
- Missing values
- Duplicates
- IDs
- Customer references
- Cross-file consistency

Resource limits are also enforced:

| Limit | Maximum |
|---|---:|
| File size | 50 MB |
| Rows per file | 500,000 |
| Files per upload | 4 |

User-uploaded data is excluded from Git version control.

---

## Feature Engineering

Customer-level features are calculated using information available up to the observation date.

### RFM Features

- **Recency** — Days since the customer's last purchase
- **Frequency** — Number of purchases
- **Monetary** — Total customer spending
- **Average Bill**
- **Average Purchase Gap**

### Additional Customer Features

- Customer tenure
- Campaigns sent
- Campaigns delivered
- Campaigns clicked
- Campaigns redeemed
- Delivery rate
- Click rate
- Redemption rate
- Previous redemptions

Future information is excluded from feature calculations to reduce data leakage.

---

## Churn Definition

Churn is defined using a **90-day future purchase window**.

```text
churn = 1  →  No purchase during the following 90 days
churn = 0  →  At least one purchase during the following 90 days
```

This creates a supervised learning target from historical customer observations.

---

## Machine Learning

Three supervised models are trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation uses:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Feature Importance

### Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 77.87% | 42.51% | 68.16% | 52.36% | 0.8172 |
| Decision Tree | 83.55% | 53.72% | 56.42% | 55.04% | 0.7293 |
| **Random Forest** | **89.33%** | **86.00%** | **48.04%** | **61.65%** | **0.8639** |

The Random Forest is selected based on the highest ROC-AUC among the evaluated models.

### Random Forest Test Results

```text
True Negatives   : 810
False Positives  : 14
False Negatives  : 93
True Positives   : 86
```

---

## Churn Risk Classification

Predicted churn probabilities are converted into four business-facing risk levels:

| Churn Probability | Risk Level |
|---:|---|
| `< 0.25` | Low |
| `0.25 – < 0.50` | Medium |
| `0.50 – < 0.75` | High |
| `≥ 0.75` | Very High |

These thresholds are operational classification rules, not probabilities of guaranteed customer behavior.

---

## Customer Segmentation

RFM analysis produces the following customer segments:

- Champions
- Loyal Customers
- Potential Loyalists
- Regular Customers
- At Risk
- Inactive
- Lost

Segmentation is used alongside churn risk to provide additional customer context rather than replacing the predictive model.

---

## Retention Priority

The system combines predicted churn risk and customer value into a **0–100 Retention Priority Score**.

```text
Retention Priority
= 60% × predicted churn probability
+ 40% × customer value percentile
```

Priority levels:

| Score | Priority |
|---:|---|
| `< 25` | Low |
| `25 – < 50` | Medium |
| `50 – < 75` | High |
| `≥ 75` | Critical |

### Expected Revenue at Risk

```text
Expected Revenue at Risk
= churn probability × customer monetary value
```

This is an analytical exposure metric. It is **not a forecast of actual future revenue loss** and does not assume that retention intervention will succeed.

---

## Campaign Analytics

Campaign performance is evaluated through the funnel:

```text
Sent → Delivered → Clicked → Redeemed
```

Current analytical results:

| Metric | Value |
|---|---:|
| Sent | 8,000 |
| Delivered | 7,311 |
| Clicked | 2,236 |
| Redeemed | 701 |
| Delivery Rate | 91.39% |
| Click Rate | 30.58% |
| Redemption Rate | 9.59% |

Campaign performance can also be compared by campaign type.

---

## Campaign Affinity

Campaign Affinity evaluates **observed historical customer response by campaign type**.

The analysis uses customer × campaign-type delivered and clicked activity to calculate response metrics such as click rate.

Current analysis:

```text
Customer × Campaign Type rows : 4,421
Unique customers              : 999
Campaign types                : 6
Delivered exposures           : 7,311
Clicks                        : 2,236
```

Campaign Affinity is descriptive. Historical response does **not** establish:

- causal campaign effectiveness
- customer preference
- guaranteed future response

---

## Feature Importance

The application exposes ranked **Random Forest feature importance** to show which features contribute most strongly to the model's tree-based predictive signal.

Feature importance is treated as descriptive model information, not as a causal explanation of customer churn.

---

## Streamlit Dashboard

The application is organized into 8 views:

| # | View | Purpose |
|---:|---|---|
| 1 | **Executive Overview** | High-level customer, churn, and campaign KPIs |
| 2 | **Data & Upload** | Controlled CSV upload and dataset handling |
| 3 | **Data Quality** | Validation results and data-quality diagnostics |
| 4 | **Customer Search** | Customer-level exploration and profile lookup |
| 5 | **Customer Intelligence** | Customer behavior, segments, and engagement insights |
| 6 | **Retention & Model Insights** | Churn predictions, risk, priority, and model information |
| 7 | **Campaign Performance** | Campaign funnel and performance analysis |
| 8 | **Campaign Affinity** | Customer-level response analysis by campaign type |

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   ├── app.py
│   ├── campaign_affinity_page.py
│   ├── customer_search_page.py
│   ├── data_quality_page.py
│   └── data_upload_page.py
│
├── data/
│   └── training/
│       ├── campaigns.csv
│       ├── customer_features.csv
│       ├── customers.csv
│       └── transactions.csv
│
├── models/
│   ├── decision_tree.joblib
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── src/
│   ├── analytics/
│   │   ├── campaign.py
│   │   ├── campaign_affinity.py
│   │   ├── customer.py
│   │   ├── customer_search.py
│   │   ├── data_quality.py
│   │   ├── retention.py
│   │   └── segmentation.py
│   │
│   ├── data/
│   │   ├── custom_pipeline.py
│   │   ├── loader.py
│   │   ├── schema.py
│   │   ├── templates.py
│   │   ├── upload.py
│   │   └── validator.py
│   │
│   ├── features/
│   │   ├── churn_label.py
│   │   └── feature_engineering.py
│   │
│   ├── ml/
│   │   ├── evaluate.py
│   │   ├── feature_importance.py
│   │   ├── predict.py
│   │   ├── prepare_dataset.py
│   │   ├── risk.py
│   │   └── train.py
│   │
│   ├── preprocessing/
│   │   └── preprocessing.py
│   │
│   └── visualization/
│       └── charts.py
│
├── tests/
│   ├── test_campaign_affinity.py
│   ├── test_campaign_features.py
│   ├── test_churn_label.py
│   ├── test_evaluate.py
│   ├── test_feature_importance.py
│   ├── test_full_preprocessing.py
│   ├── test_model_persistence.py
│   ├── test_predict.py
│   ├── test_retention.py
│   ├── test_rfm.py
│   ├── test_risk.py
│   ├── test_segmentation.py
│   ├── test_train.py
│   ├── test_upload.py
│   ├── test_upload_limits.py
│   ├── test_upload_validation.py
│   └── ...
│
├── .gitignore
├── config.py
├── README.md
└── requirements.txt
```

---

## Testing

The project includes an automated test suite covering data, analytics, machine learning, dashboard logic, and integration behavior.

### Current Test Status

**125 tests passing**

Coverage includes:

- Data validation
- Upload validation
- Upload resource limits
- Preprocessing
- Date conversion
- Numeric conversion
- Missing-value checks
- Duplicate detection
- Customer references
- Customer tenure
- RFM features
- Campaign features
- Previous redemptions
- Customer feature generation
- Churn labeling
- Dataset preparation
- Model training
- Model evaluation
- Model persistence
- Prediction
- Risk classification
- Segmentation
- Retention priority
- Campaign analytics
- Campaign Affinity
- Feature importance
- Reusable visualizations
- Customer Search
- Data Quality dashboard logic
- Streamlit page behavior

Run the complete test suite with:

```bash
pytest -q
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## Analytical Limitations

The current system is intentionally positioned as a decision-support prototype.

### Churn prediction

The model learns from historical customer behavior and should not be interpreted as a guarantee of future customer actions.

### Probability interpretation

Predicted probabilities are model outputs. Probability calibration and threshold optimization are areas for further analytical development.

### Campaign Affinity

Observed campaign response does not establish causal campaign effectiveness or individual customer preference.

### Retention value

Expected Revenue at Risk represents potential economic exposure based on predicted churn probability and customer monetary value. It does not represent revenue that will definitely be lost or saved.

### Intervention impact

The system does not currently estimate treatment effects, uplift, or incremental revenue generated by a specific retention action.

---

## Current Project Status

### Completed

- Data validation
- Controlled CSV upload
- Resource limits
- Data preprocessing
- Customer-level feature engineering
- RFM analysis
- Customer segmentation
- 90-day churn labeling
- Logistic Regression, Decision Tree, and Random Forest modeling
- Model evaluation and persistence
- Churn probability prediction
- Risk classification
- Retention Priority
- Expected Revenue at Risk
- Campaign funnel analytics
- Campaign Affinity
- Random Forest feature importance
- Customer Search
- Data Quality analysis
- Reusable visualizations
- Streamlit dashboard with 8 views
- Automated test suite with 125 passing tests
- Project documentation

### Next Analytical Development

Planned analytical enhancements focus on improving model and business decision quality:

1. Churn Threshold Sensitivity
2. Calibration / Model Reliability
3. Failure Mode Analysis
4. Segment-Level Model Performance
5. Economic Value-at-Risk
6. Risk vs. Economic Value Targeting Comparison

---

## Documentation

The complete technical project documentation covers:

- Data schemas and validation
- Preprocessing
- Feature engineering
- Churn labeling
- Machine-learning methodology
- Model evaluation
- Prediction and risk scoring
- RFM segmentation
- Retention prioritization
- Campaign analytics
- Campaign Affinity
- Feature importance
- Streamlit application
- Testing and verification
- Scope and limitations

---

## Repository

**GitHub:**  
https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction
