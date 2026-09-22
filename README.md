Customer Churn Prediction & Campaign Analytics System

An end-to-end machine-learning customer intelligence and churn decision-support system for understanding customer behavior, predicting churn risk, prioritizing retention opportunities, analyzing campaign performance, and quantifying customer-level economic exposure.

Built with: Python · Pandas · NumPy · scikit-learn · Plotly · Streamlit · pytest

Overview

This project combines customer profiles, transaction history, campaign activity, machine-learning predictions, RFM segmentation, retention prioritization, campaign analytics, customer value, retention cost, and economic risk analysis into a single analytical application.

The workflow runs from controlled CSV ingestion through validation, preprocessing, feature engineering, churn modeling, customer intelligence, campaign analysis, economic analysis, and interactive Streamlit decision support.

Project Positioning

This is a portfolio-oriented decision-support system.

It is not:

an autonomous campaign execution platform

a production deployment

a treatment-effect or uplift modeling system

a guarantee of future customer behavior or revenue outcomes

Key Capabilities

Capability

Description

Controlled Data Upload

Accepts customer, transaction, and campaign CSV data through a validated upload workflow

Data Quality

Checks schemas, data types, dates, missing values, duplicates, IDs, customer references, and cross-file consistency

Feature Engineering

Builds customer-level behavioral, transaction, RFM, and campaign features

90-Day Churn Prediction

Predicts the probability that a customer will fail to make a purchase during the following 90 days

Risk Classification

Converts churn probability into Low, Medium, High, and Very High risk levels

RFM Segmentation

Classifies customers into behavioral segments using Recency, Frequency, and Monetary value

Retention Priority

Combines churn probability and relative customer value into a 0–100 priority score

Expected Revenue at Risk

Calculates probability-weighted customer value exposed to predicted churn

Campaign Analytics

Measures the Sent → Delivered → Clicked → Redeemed campaign funnel

Campaign Affinity

Analyzes observed customer response across campaign types

Feature Importance

Provides Random Forest feature importance as descriptive model information

Customer Search

Supports customer-level profile and behavioral exploration

Customer Value

Calculates historical customer value from transaction activity

Retention Cost

Calculates observed historical campaign-based retention cost

Economic Value at Risk

Combines churn probability and historical customer value into probability-weighted economic exposure

Economic Priority

Ranks customers using cost-adjusted economic exposure

Model Diagnostics

Includes churn-threshold sensitivity and segment-level model performance auditing

Custom Model Evaluation

Supports evaluation of supplied/custom customer data through the model-diagnostics workflow

Interactive Dashboard

Provides 9 functional Streamlit views across data, model quality, analytics, and retention

System Architecture

                    Customer CSV Data
                           │
                           ▼
                ┌─────────────────────┐
                │ Schema Validation   │
                │ Data Quality Checks │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Preprocessing    │
                │ Types / Dates / IDs │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering │
                │ RFM + Behaviour     │
                │ Campaign Features   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ 90-Day Churn Label  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Machine Learning    │
                │ LR / DT / RF        │
                └──────────┬──────────┘
                           │
                           ▼
          ┌──────────────────────────────────┐
          │       Customer Intelligence     │
          │ Probability / Risk / RFM         │
          │ Segmentation / Retention Priority│
          └───────────────┬──────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
   ┌────────────────────┐  ┌────────────────────┐
   │ Campaign Analytics │  │ Model Diagnostics  │
   │ Funnel / Affinity  │  │ Threshold / Audit  │
   └──────────┬─────────┘  └────────────────────┘
              │
              ▼
   ┌────────────────────────────┐
   │ Economic Intelligence     │
   │ Customer Value             │
   │ Retention Cost             │
   │ EVaR / Net Value at Risk   │
   │ Economic Priority          │
   └────────────┬───────────────┘
                │
                ▼
   ┌────────────────────────────┐
   │ Streamlit Decision Support │
   └────────────────────────────┘

Data & Validation

The application works with three primary raw datasets.

Customers

customer_id
gender
age
city
registration_date

Transactions

transaction_id
customer_id
transaction_date
bill_amount
outlet

Campaigns

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

Derived Customer Features

customer_features.csv is a derived customer-level dataset used for machine-learning preparation and reference. It is not one of the three raw upload schemas.

Additional Campaign Analytics Data

The repository also contains:

data/analytics/campaign_outcomes.csv

data/analytics/synthetic_campaign_experiment.csv

campaign_outcomes.csv contains historical campaign/customer exposures with post-campaign purchase and revenue outcomes at 7-, 14-, and 30-day windows.

synthetic_campaign_experiment.csv is synthetic randomized analytical data used for experimentation. It must not be presented as real-world historical evidence.

Validation Controls

Uploaded data is checked for:

required columns

unexpected columns

data types

date validity

missing values

duplicates

identifier validity

customer references

cross-file consistency

file-size limits

row-count limits

upload-count limits

Configured resource limits include:

Limit

Maximum

File size

50 MB

Rows per file

500,000

Files per upload

4

Feature Engineering

Customer-level features are calculated using information available up to the observation date.

RFM Features

Recency — days since the customer's last purchase

Frequency — number of purchases

Monetary — total customer spending

Average bill

Average purchase gap

Additional Features

Customer tenure

Campaigns sent

Campaigns delivered

Campaigns clicked

Campaigns redeemed

Delivery rate

Click rate

Redemption rate

Previous redemptions

Future information is excluded from feature calculations to reduce data leakage.

Churn Definition

Churn is defined using a 90-day future purchase window.

churn = 1  →  No purchase during the following 90 days
churn = 0  →  At least one purchase during the following 90 days

The model therefore estimates the probability that, given information available at an observation date, a customer will fail to make a purchase during the next 90 days.

Machine Learning

Three supervised models are evaluated:

Logistic Regression

Decision Tree

Random Forest

Evaluation includes:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Feature Importance

Model Performance

Model

Accuracy

Precision

Recall

F1

ROC-AUC

Logistic Regression

77.87%

42.51%

68.16%

52.36%

0.8172

Decision Tree

83.55%

53.72%

56.42%

55.04%

0.7293

Random Forest

89.33%

86.00%

48.04%

61.65%

0.8639

The Random Forest is selected based on the highest ROC-AUC among the evaluated models.

Random Forest Confusion Matrix

True Negatives   : 810
False Positives  : 14
False Negatives  : 93
True Positives   : 86

Churn Risk Classification

Predicted churn probabilities are converted into four business-facing risk levels:

Churn Probability

Risk Level

< 0.25

Low

0.25 – < 0.50

Medium

0.50 – < 0.75

High

≥ 0.75

Very High

These thresholds are operational classification rules. They are not guarantees of customer behavior.

Customer Segmentation

RFM analysis produces seven customer segments:

Champions

Loyal Customers

Potential Loyalists

Regular Customers

At Risk

Inactive

Lost

Segmentation provides behavioral context alongside the churn model. It does not replace the predictive model.

Retention Priority

The system combines predicted churn risk and relative customer value into a 0–100 Retention Priority Score.

Retention Priority
= 60% × predicted churn probability
+ 40% × customer value percentile

Priority levels:

Score

Priority

< 25

Low

25 – < 50

Medium

50 – < 75

High

≥ 75

Critical

The system also calculates:

Expected Revenue at Risk
= churn probability × historical customer value

Expected Revenue at Risk is probability-weighted exposure, not guaranteed future revenue loss.

Campaign Analytics

Campaign performance is evaluated through:

Sent → Delivered → Clicked → Redeemed

The current analytical dataset reports:

Metric

Value

Sent

8,000

Delivered

7,311

Clicked

2,236

Redeemed

701

Delivery Rate

91.39%

Click Rate

30.58%

Redemption Rate

9.59%

Campaign performance can also be examined by campaign type.

Campaign Affinity

Campaign Affinity evaluates observed customer response by campaign type.

The analysis uses customer × campaign-type delivery and engagement activity to calculate descriptive response metrics such as click rate.

Current analysis includes:

Customer × Campaign Type rows : 4,421
Unique customers              : 999
Campaign types                : 6
Delivered exposures           : 7,311
Clicks                        : 2,236

Campaign Affinity is descriptive.

It does not establish:

causal campaign effectiveness

individual customer preference

guaranteed future response

incremental revenue from a campaign

Economic Intelligence

The economic layer extends the churn model into customer-level economic analysis.

Customer Value

Customer Value represents historical monetary contribution based on transaction activity.

The implementation includes measures such as:

total historical revenue

purchase count

average order value

active days

annualized revenue

customer value

customer_value is a historical metric, not a predictive lifetime-value model.

Retention Cost

Retention Cost uses observed campaign cost information to quantify historical campaign expenditure.

It supports:

campaign-level retention cost

customer-level retention cost

total retention cost

average retention cost

retention cost by campaign type

Historical campaign cost is treated as an observed cost measure. It is not automatically interpreted as the cost of a future intervention.

Expected Value at Risk

Expected Value at Risk
= churn probability × customer value

Example:

Customer value = ₹20,000
Churn probability = 70%

EVaR = ₹20,000 × 0.70
     = ₹14,000

EVaR represents probability-weighted economic exposure.

It does not mean that ₹14,000 will definitely be lost, saved, or recovered.

Net Value at Risk

Net Value at Risk
= Expected Value at Risk − Retention Cost

This provides a cost-adjusted economic exposure measure used by the Economic Priority analysis.

Economic Priority

Economic Priority ranks customers using Net Value at Risk.

The system classifies customers into:

Low

Medium

High

Critical

The purpose is to provide a customer-level economic prioritization layer on top of churn risk.

Model Diagnostics

The project includes a dedicated Model Diagnostics & Threshold Analysis view.

Threshold Sensitivity

The diagnostic evaluates model classification behavior across probability thresholds rather than treating the default threshold as universally optimal.

The interface compares:

Precision

Recall

F1-score

across tested churn-probability thresholds.

The existing 50% threshold is displayed as the baseline.

The analysis reuses existing model probabilities and held-out labels; it does not retrain the model.

Segment-Level Performance Audit

Model performance can be examined across RFM customer segments using metrics such as:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Sample count

These results describe model behavior within the supplied evaluation data. They are diagnostic observations, not causal segment effects.

Additional Analytical Modules

The backend also contains tested analytical modules for:

Model Calibration / Reliability

Failure Mode Analysis

Segment-Level Model Performance Audit

Churn Threshold Sensitivity

Custom Model Evaluation

These modules are kept separate from the core prediction pipeline so that model diagnostics do not alter the production prediction logic.

Streamlit Dashboard

The final stable application contains 9 functional views, excluding the Home Page:

View

Purpose

Data & Upload

Controlled CSV ingestion and active dataset management

Customer Search

Customer-level exploration and profile lookup

Model Diagnostics

Threshold sensitivity and segment-level model auditing

Data Quality

Dataset validation and quality diagnostics

Customer Intelligence

Customer behavior, segmentation, and engagement analysis

Retention & Model Insights

Churn drivers, retention priority, and customer risk insights

Campaign Performance

Campaign funnel and campaign-type performance

Campaign Affinity

Customer-level campaign response analysis

Economic Intelligence

Customer value, retention cost, EVaR, and economic exposure

The application also includes a dedicated Home Page for navigation and overview.

UI Design

The dashboard uses a unified visual system built around:

Dark-blue application shell

Compact sidebar navigation

Warm cream information cards

Consistent KPI cards

Consistent analytical tables

Plotly-based analytical charts

Dedicated model-quality and economic views

Responsive Streamlit layouts

The primary application palette is centered around:

#0A2947
#D3D4C0
#F3E4C9
#8B5E3C
#778873

Project Structure

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
│   ├── data_upload_page.py
│   ├── economic_pages.py
│   └── model_diagnostics_page.py
│
├── data/
│   ├── analytics/
│   │   ├── campaign_outcomes.csv
│   │   └── synthetic_campaign_experiment.csv
│   │
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
│   │   ├── churn_threshold.py
│   │   ├── customer.py
│   │   ├── customer_search.py
│   │   ├── customer_value.py
│   │   ├── custom_model_evaluation.py
│   │   ├── data_quality.py
│   │   ├── economic_priority.py
│   │   ├── economic_value.py
│   │   ├── failure_modes.py
│   │   ├── model_calibration.py
│   │   ├── retention.py
│   │   ├── retention_cost.py
│   │   ├── segment_model_audit.py
│   │   └── segmentation.py
│   │
│   ├── data/
│   │   ├── analytics_loader.py
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
│   └── comprehensive pytest suite
│
├── .gitignore
├── config.py
├── README.md
└── requirements.txt

Testing

The repository includes a comprehensive automated test suite covering:

Data validation

Upload validation

Upload resource limits

Schema validation

Preprocessing

Date conversion

Numeric conversion

Missing-value checks

Duplicate detection

Customer references

Customer tenure

RFM features

Campaign features

Previous redemptions

Customer feature generation

Churn labeling

Dataset preparation

Model training

Model evaluation

Model persistence

Prediction

Risk classification

Customer segmentation

Retention Priority

Customer Search

Data Quality

Campaign Analytics

Campaign Affinity

Feature Importance

Customer Value

Retention Cost

Economic Value at Risk

Economic Priority

Churn Threshold Sensitivity

Model Calibration / Reliability

Failure Mode Analysis

Segment-Level Model Performance Audit

Custom Model Evaluation

Reusable visualizations

Streamlit page behavior

Run the complete suite with:

pytest -q

The repository's documented development checkpoint recorded 201 passing automated tests before the later Retention Planning work was attempted.

Installation

1. Clone the repository

git clone https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Create a virtual environment

python -m venv .venv

3. Activate it on Windows PowerShell

.venv\Scripts\Activate.ps1

4. Install dependencies

pip install -r requirements.txt

5. Run the Streamlit application

streamlit run app/app.py

The application will open in your browser.

Analytical Limitations

The system is intentionally positioned as a decision-support application.

Churn Prediction

The model learns from historical customer behavior. A predicted churn probability is not a guarantee that a customer will churn.

Probability Interpretation

Predicted probabilities are model outputs. They should not automatically be interpreted as perfectly calibrated real-world probabilities.

Feature Importance

Random Forest feature importance describes the contribution of features to model behavior. It does not establish causal drivers of churn.

RFM Segmentation

RFM segments describe historical customer behavior. They do not represent immutable customer identities or future outcomes.

Retention Priority

Retention Priority is a business scoring framework based on churn probability and relative customer value. It is a prioritization tool, not a prediction of intervention success.

Expected Revenue at Risk / EVaR

Expected Revenue at Risk and EVaR represent probability-weighted economic exposure. They do not represent guaranteed revenue loss, guaranteed savings, or guaranteed recovery.

Retention Cost

Historical campaign cost is an observed cost measure. It should not automatically be interpreted as the cost of a future intervention.

Campaign Affinity

Observed campaign response does not establish customer preference or causal campaign effectiveness.

Campaign Outcomes

Historical campaign outcomes are observational. They should not be interpreted as randomized treatment effects.

Synthetic Experiment

The synthetic campaign experiment is generated analytical data. It must not be presented as real customer evidence or real-world causal evidence.

Intervention Impact

The project does not estimate treatment effects, uplift, or guaranteed incremental revenue from a particular retention action.

Final Project Scope

This repository represents the final stable project checkpoint before the Retention Planning / Page 2 experiment.

The completed scope ends with:

Data & Upload
      ↓
Data Quality
      ↓
Customer Intelligence
      ↓
Churn Prediction & Risk
      ↓
RFM Segmentation
      ↓
Retention Priority
      ↓
Campaign Performance
      ↓
Campaign Affinity
      ↓
Model Diagnostics
      ↓
Economic Intelligence

The following Retention Planning concepts were explored after this checkpoint but are not part of the final documented application scope:

Retention Budget Simulator

Campaign Portfolio Simulator

Budget-Constrained Portfolio Optimization

They are intentionally excluded from this README so the repository documentation reflects the stable project rather than an unfinished experimental extension.

Documentation

The repository contains the project's technical documentation under:

docs/
├── PROJECT DOCUMENTATION.docx
└── PROJECT_DOCUMENTATION.docx

These documents contain the detailed implementation history, methodology, testing evidence, screenshots, and technical notes.

GitHub

Repository:
https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction

Project Summary

Customer Churn Prediction & Campaign Analytics System is an end-to-end machine-learning customer intelligence project combining:

Customer Data
     +
Transaction Behaviour
     +
Campaign Activity
     +
Machine Learning
     +
RFM Segmentation
     +
Retention Prioritization
     +
Campaign Analytics
     +
Economic Intelligence
     =
Decision-Support System

It demonstrates how a churn model can be extended beyond a simple prediction into a broader analytical system for understanding customers, evaluating risk, prioritizing retention opportunities, analyzing campaign behavior, and quantifying economic exposure.

