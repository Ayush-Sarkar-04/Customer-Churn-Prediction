Customer Churn Prediction and Campaign Analytics System

An end-to-end customer intelligence and churn decision-support system built with Python, scikit-learn, Pandas, Plotly, and Streamlit.

The system combines customer profiles, transaction history, campaign activity, machine-learning predictions, RFM segmentation, retention prioritization, campaign analytics, customer-level campaign affinity, and model feature importance into an integrated analytical application.

Project positioning: This is a portfolio-oriented decision-support system, not an autonomous campaign execution or production deployment platform.

Project Overview

The system is designed to answer practical customer-retention and campaign-analytics questions:

Is the input data trustworthy?

Which customers are most likely to churn?

Which customers deserve the highest retention attention?

Which customer segments require intervention?

Which campaign types perform best overall?

Which campaign types generate the strongest observed click response for an individual customer?

Which features contribute most to the Random Forest churn prediction?

The pipeline runs from controlled CSV ingestion through validation, preprocessing, feature engineering, machine-learning prediction, segmentation, campaign analytics, and Streamlit-based decision support.

Key Capabilities

Data Quality and Controlled Uploads

Fixed-schema validation for Customers, Transactions, and Campaigns

Data-type, date, missing-value, duplicate, ID, and cross-file reference validation

Controlled custom CSV upload

19 user-facing data-quality checks

Resource protection: 50 MB maximum file size, 500,000 maximum rows per file, 4 maximum uploaded files

User uploads excluded from Git version control

Customer-Level Feature Engineering

The system generates customer-level behavioral features using information available up to the observation date.

Purchase and RFM features: Recency, Frequency, Monetary, Average Bill, Average Purchase Gap, Customer Tenure.

Campaign engagement: Campaigns Sent, Delivered, Clicked, Redeemed, Delivery Rate, Click Rate, Redemption Rate, Previous Redemptions.

Future information is excluded from feature calculations to reduce data leakage.

Churn Definition

A customer is considered churned when they make no purchase during the following 90 days:

churn = 1 → No purchase in future 90 days
churn = 0 → At least one purchase in future 90 days

The supervised ML dataset contains 5,015 customer-level observations.

Machine Learning

Three supervised models are trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

Evaluation includes Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix.

Model

Accuracy

Precision

Recall

F1

ROC-AUC

Logistic Regression

0.7787

0.4251

0.6816

0.5236

0.8172

Decision Tree

0.8355

0.5372

0.5642

0.5504

0.7293

Random Forest

0.8933

0.8600

0.4804

0.6165

0.8639

Random Forest is the current prediction model because it achieved the highest ROC-AUC.

Random Forest evaluation confusion matrix:

TN = 810    FP = 14
FN = 93     TP = 86

Churn Probability and Risk Classification

Risk Level

Churn Probability

Low

< 0.25

Medium

0.25 – < 0.50

High

0.50 – < 0.75

Very High

≥ 0.75

Binary churn prediction and business-facing risk classification are separate outputs.

RFM Customer Segmentation

Seven RFM-based segments are implemented:

Champions

Loyal Customers

Potential Loyalists

Regular Customers

At Risk

Inactive

Lost

Segmentation is integrated with churn predictions and risk levels for customer-level analysis.

Retention Priority and Expected Revenue at Risk

The system calculates a 0–100 Retention Priority Score using predicted churn probability and customer-value percentile, with priority bands from Low through Critical.

It also calculates Expected Revenue at Risk as a decision-support exposure metric.

Expected Revenue at Risk is not a forecast of revenue that will definitely be lost.

Campaign Analytics

Campaign performance is analyzed through:

Sent → Delivered → Clicked → Redeemed

Current overall campaign results:

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

Campaign performance is also compared by campaign type.

Campaign Affinity

Campaign Affinity evaluates customer-level observed response by campaign type using delivered and clicked activity.

It provides:

Customer × Campaign Type response

Delivered exposure

Clicks

Click rate

Best Campaign Type

Campaign Affinity describes observed response, not proven customer preference or causal campaign effectiveness.

Random Forest Feature Importance

The application exposes ranked Random Forest feature importance to show which features contribute most strongly to the model's tree-based predictive signal.

Feature importance is treated as descriptive model signal, not causal explanation.

Streamlit Application

The current application contains eight integrated views:

Executive Overview

Data & Upload

Data Quality

Customer Search

Customer Intelligence

Retention & Model Insights

Campaign Performance

Campaign Affinity

Application Architecture

Validated Demo Data / Custom CSV Upload
                ↓
      Resource Protection
                ↓
 Fixed-Schema & Data Validation
                ↓
 Cross-File Reference Validation
                ↓
          Preprocessing
                ↓
       Feature Engineering
                ↓
  90-Day Churn Labeling / ML Dataset
                ↓
     Model Training & Evaluation
                ↓
        Model Persistence
                ↓
 Churn Prediction & Probability
                ↓
       Risk Classification
                ↓
      RFM Segmentation
                ↓
   Combined Customer Analytics
                ↓
 Retention Priority / Revenue at Risk
                ↓
     Campaign Performance
                ↓
       Campaign Affinity
                ↓
   Random Forest Feature Importance
                ↓
       Reusable Visualizations
                ↓
     Streamlit Decision Support

Project Structure

Customer-Churn-Prediction/
├── .streamlit/
├── app/
├── data/
│   └── training/
├── models/
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

Raw Dataset Schemas

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

customer_features.csv is a derived customer-level dataset used for machine-learning preparation and reference. It is not one of the three raw upload schemas.

Testing

The automated test suite covers validation, preprocessing, feature engineering, churn labeling, model training/evaluation/persistence, prediction, risk classification, segmentation, customer analytics, visualization, Data Quality, Customer Search, Retention Priority, Feature Importance, Campaign Analytics, and Campaign Affinity.

Current Test Result

125 automated tests passing

Focused evidence includes:

Feature Importance: 9 tests

Campaign Affinity: 11 tests

Visualization: 6 tests

Customer Search backend: 8 tests

Customer Search Streamlit page: 3 tests

Data Quality validation: 4 tests

Data Quality Streamlit page: 1 test

Customer analytics regression: 5 tests

Technology Stack

Python

Pandas

NumPy

scikit-learn

Plotly

Matplotlib

Joblib

Streamlit

pytest

openpyxl

Git / GitHub

Installation

git clone https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m venv .venv

Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app/app.py

Analytical Limitations

Retention Priority is a prioritization proxy.

Expected Revenue at Risk is not a true revenue forecast.

Random Forest feature importance is not a causal explanation.

Campaign Affinity describes observed click response rather than proving customer preference.

Historical campaign response does not establish causal campaign effectiveness.

The current system does not claim that a particular retention intervention will prevent churn.

Current Project Status

The core system is implemented and integrated, including validation and resource protection, preprocessing, feature engineering, 90-day churn labeling, ML training/evaluation, model persistence, churn prediction and risk classification, RFM segmentation, customer analytics, Retention Priority, Expected Revenue at Risk, campaign analytics, Campaign Affinity, Random Forest Feature Importance, reusable visualizations, Customer Search, Data Quality, controlled custom CSV upload, and the eight-view Streamlit application.

The project remains a portfolio system rather than a production deployment. Authentication/authorization, operational monitoring, model/version governance, data-drift monitoring, deployment hardening, and true request-frequency rate limiting are outside the current scope.

Documentation

The complete system documentation contains the detailed implementation, methodology, validation logic, model evaluation, feature engineering, application views, test evidence, screenshots, and analytical limitations.

Repository

GitHub Repository
