Customer Churn Prediction & Campaign Analytics System

An end-to-end customer intelligence and churn decision-support system for understanding customer behavior, predicting churn, prioritizing retention opportunities, and analyzing campaign performance.

Built with: Python · Pandas · NumPy · scikit-learn · Plotly · Streamlit

Overview

This project combines customer profiles, transaction history, campaign activity, machine-learning predictions, RFM segmentation, retention prioritization, campaign analytics, and customer-level campaign affinity into an integrated analytical application.

The workflow runs from controlled CSV ingestion → validation → feature engineering → churn modeling → customer intelligence → campaign analytics → interactive Streamlit decision support.

Project positioning: This is a portfolio-oriented decision-support system, not an autonomous campaign execution or production deployment platform.

What the System Does

Capability

Purpose

Data Quality

Validate customer, transaction, and campaign data before analysis

Customer Features

Build customer-level behavioral and engagement features

Churn Prediction

Estimate the probability of churn using supervised ML

Risk Classification

Convert churn probability into business-facing risk levels

RFM Segmentation

Group customers by purchasing behavior and value

Retention Priority

Rank customers using churn risk and customer value

Revenue at Risk

Estimate customer-value exposure associated with churn probability

Campaign Analytics

Analyze the Sent → Delivered → Clicked → Redeemed funnel

Campaign Affinity

Analyze observed customer response by campaign type

Model Insights

Expose Random Forest feature importance

Customer Search

Explore individual customer profiles and analytics

Project at a Glance

Metric

Current Result

ML observations

5,015

Unique current customer profiles

945

Current model

Random Forest

Random Forest ROC-AUC

0.8639

Streamlit views

8

Automated tests

125 passing

Campaigns sent

8,000

Campaigns delivered

7,311

Campaign clicks

2,236

Campaign redemptions

701

Data Quality & Controlled Uploads

Uploaded CSV files are validated against predefined schemas before being used by the analytical pipeline.

Validation covers:

Required and unexpected columns

Data types

Dates

Missing values

Duplicates

IDs

Customer references

Cross-file consistency

Resource Protection

Limit

Maximum

File size

50 MB

Rows per file

500,000

Uploaded files

4

User-uploaded files are excluded from Git version control.

Customer-Level Feature Engineering

Features are generated using information available up to the relevant observation date.

Purchase & RFM Features

Recency — Days since last purchase

Frequency — Number of purchases

Monetary — Total spending

Average Bill

Average Purchase Gap

Customer Tenure

Campaign Engagement Features

Campaigns Sent

Campaigns Delivered

Campaigns Clicked

Campaigns Redeemed

Delivery Rate

Click Rate

Redemption Rate

Previous Redemptions

Future information is excluded from feature calculations to reduce data leakage.

Churn Definition

A customer is considered churned when they make no purchase during the following 90 days.

churn = 1  →  No purchase in future 90 days
churn = 0  →  At least one purchase in future 90 days

The resulting supervised machine-learning dataset contains 5,015 customer-level observations.

Machine Learning

Three supervised models are trained and evaluated:

Logistic Regression

Decision Tree

Random Forest

Evaluation includes:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

Model Performance

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

Random Forest Test-Set Confusion Matrix



Predicted Non-Churn

Predicted Churn

Actual Non-Churn

810

14

Actual Churn

93

86

Churn Probability & Risk

Predicted churn probability and business-facing risk classification are separate outputs.

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

RFM Customer Segmentation

The system implements seven RFM-based customer segments:

Champions · Loyal Customers · Potential Loyalists · Regular Customers · At Risk · Inactive · Lost

Segmentation is integrated with churn predictions, risk levels, and customer analytics.

Retention Prioritization

The system calculates a 0–100 Retention Priority Score using:

Predicted churn probability

Customer-value percentile

Priority bands:

Low · Medium · High · Critical

Expected Revenue at Risk

The system also calculates:

Expected Revenue at Risk
= Churn Probability × Customer Monetary Value

This is a decision-support exposure metric, not a forecast of revenue that will definitely be lost.

Campaign Analytics

Campaign performance is analyzed through the funnel:

Sent → Delivered → Clicked → Redeemed

Current Campaign Results

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

Campaign performance can also be compared by campaign type.

Campaign Affinity

Campaign Affinity evaluates customer-level observed response by campaign type.

For each customer and campaign type, the application provides:

Delivered exposure

Clicks

Click rate

Best observed campaign type

This supports questions such as:

Which campaign types have historically generated the strongest observed click response for a particular customer?

Campaign Affinity describes observed historical response. It does not prove customer preference or causal campaign effectiveness.

Random Forest Feature Importance

The application exposes ranked Random Forest feature importance to show which features contribute most strongly to the model's tree-based predictive signal.

Feature importance is treated as descriptive model information, not a causal explanation.

Streamlit Application

The application contains eight integrated views:

View

Purpose

Executive Overview

High-level customer, churn, and campaign KPIs

Data & Upload

Dataset inspection and controlled CSV upload

Data Quality

Validation results and diagnostics

Customer Search

Search and inspect individual customer profiles

Customer Intelligence

Customer behavior, segmentation, and engagement

Retention & Model Insights

Churn probability, risk, and retention priority

Campaign Performance

Campaign funnel and campaign-type performance

Campaign Affinity

Customer-level campaign response

Application Architecture

Customer / Transaction / Campaign Data
                    │
                    ▼
          Resource Protection
                    │
                    ▼
       Fixed-Schema & Data Validation
                    │
                    ▼
        Cross-File Validation
                    │
                    ▼
             Preprocessing
                    │
                    ▼
          Feature Engineering
                    │
                    ▼
       90-Day Churn Labeling
                    │
                    ▼
       ML Training & Evaluation
                    │
                    ▼
          Model Persistence
                    │
                    ▼
        Churn Prediction
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   Risk Classification    RFM Segmentation
          │                   │
          └─────────┬─────────┘
                    ▼
        Combined Customer Analytics
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 Retention Priority      Revenue at Risk
          │                   │
          └─────────┬─────────┘
                    ▼
       Campaign Performance
                    │
                    ▼
        Campaign Affinity
                    │
                    ▼
    Random Forest Feature Importance
                    │
                    ▼
       Streamlit Decision Support

Project Structure

Customer-Churn-Prediction/
│
├── .streamlit/
├── app/
├── data/
│   └── training/
├── models/
│
├── src/
│   ├── analytics/
│   ├── data/
│   ├── features/
│   ├── ml/
│   ├── preprocessing/
│   └── visualization/
│
├── tests/
│
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

Derived Customer Features

customer_features.csv is a derived customer-level dataset used for machine-learning preparation and reference. It is not one of the three raw upload schemas.

Testing

The automated test suite covers:

Data validation

Preprocessing

Feature engineering

Churn labeling

Model training and evaluation

Model persistence

Prediction

Risk classification

Customer segmentation

Customer analytics

Retention Priority

Campaign Analytics

Campaign Affinity

Feature Importance

Visualization

Data Quality

Customer Search

Streamlit application behavior

Current Test Result

125 automated tests passing

Focused evidence includes:

Area

Tests

Feature Importance

9

Campaign Affinity

11

Visualization

6

Customer Search backend

8

Customer Search Streamlit page

3

Data Quality validation

4

Data Quality Streamlit page

1

Customer analytics regression

5

Technology Stack

Data & Analysis

Python · Pandas · NumPy

Machine Learning

scikit-learn · Joblib

Visualization

Plotly · Matplotlib

Application

Streamlit

Testing

pytest

Utilities

openpyxl

Version Control

Git · GitHub

Run Locally

1. Clone the Repository

git clone https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Create a Virtual Environment

python -m venv .venv

3. Activate the Environment

Windows PowerShell

.venv\Scripts\Activate.ps1

4. Install Dependencies

pip install -r requirements.txt

5. Launch the Application

streamlit run app/app.py

Analytical Scope & Limitations

The system is designed as a decision-support application.

It does not claim that:

predicted churn will definitely occur

Expected Revenue at Risk represents guaranteed revenue loss

campaign affinity proves customer preference

historical campaign response establishes causal campaign effectiveness

feature importance represents causal drivers

a particular retention intervention will prevent churn

The project does not currently implement autonomous campaign execution.

Documentation

The complete project documentation contains the detailed implementation and methodology, including:

Data validation

Preprocessing

Feature engineering

Churn labeling

Machine-learning methodology

Model evaluation

Model persistence

Risk classification

RFM segmentation

Retention Priority

Expected Revenue at Risk

Campaign Analytics

Campaign Affinity

Feature Importance

Streamlit application

Testing and verification

Screenshots

Analytical limitations

Project Status

Implemented

Data validation and resource protection

Controlled custom CSV upload

Data preprocessing

Customer-level feature engineering

90-day churn labeling

ML training and evaluation

Model persistence

Churn probability prediction

Risk classification

RFM segmentation

Customer analytics

Retention Priority

Expected Revenue at Risk

Campaign Analytics

Campaign Affinity

Random Forest Feature Importance

Reusable visualizations

Customer Search

Data Quality

Eight-view Streamlit application

Automated test suite with 125 passing tests

Portfolio Scope

The project is intentionally positioned as a portfolio-oriented analytical system rather than a production deployment.

The following remain outside the current scope:

Authentication and authorization

Operational monitoring

Model/version governance

Data-drift monitoring

Deployment hardening

True request-frequency rate limiting

Autonomous campaign execution

Repository

GitHub Repository

https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction