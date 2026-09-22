# Customer Churn Prediction & Campaign Analytics System
 
An end-to-end machine-learning customer intelligence and churn decision-support system for understanding customer behavior, predicting churn risk, prioritizing retention opportunities, analyzing campaign performance, and quantifying customer-level economic exposure.
 
---
 
## Built With
 
Python | Pandas | NumPy | scikit-learn | Plotly | Streamlit | pytest
 
---
 
## Overview
 
This project combines customer profiles, transaction history, campaign activity, and machine-learning predictions into a single analytical application. 
 
The workflow runs from controlled CSV ingestion through validation, preprocessing, feature engineering, churn modeling, and interactive Streamlit decision support.
 
**Project Positioning**: This is a portfolio-oriented decision-support system. It is not an autonomous campaign execution platform, production deployment, or treatment-effect model.
 
---
 
## Key Capabilities
 
**Data Management**
- Controlled CSV ingestion with schema validation
- Data quality checks (types, dates, duplicates, cross-file consistency)
- Upload resource limits and file validation
**Feature Engineering**
- Customer-level behavioral features
- RFM (Recency, Frequency, Monetary) analysis
- Transaction-based metrics
- Campaign engagement features
**Churn Prediction**
- 90-day churn probability prediction
- Risk classification (Low/Medium/High/Very High)
- Confusion matrix and ROC-AUC evaluation
**Customer Segmentation**
- 7-segment RFM segmentation
- Champions, Loyal Customers, At Risk, Lost, and more
- Retention priority scoring (0-100 scale)
**Campaign Analytics**
- Sent → Delivered → Clicked → Redeemed funnel tracking
- Campaign-type performance analysis
- Customer affinity across campaign types
**Economic Intelligence**
- Customer value calculation from transaction history
- Retention cost from campaign spend
- Expected Value at Risk (EVaR) analysis
- Net Value at Risk with cost adjustment
- Economic priority ranking
**Model Diagnostics**
- Churn threshold sensitivity analysis
- Segment-level model performance auditing
- Feature importance reporting
- Failure mode analysis
---
 
## System Architecture
 
```
Customer CSV Data
       |
       v
   Schema Validation
   Data Quality Checks
       |
       v
   Preprocessing
   (Types, Dates, IDs)
       |
       v
   Feature Engineering
   (RFM, Behaviour, Campaign)
       |
       v
   90-Day Churn Label
       |
       v
   Machine Learning
   (Logistic Regression / Decision Tree / Random Forest)
       |
       v
   Customer Intelligence Layer
   (Probability, Risk, Segmentation)
       |
       +-- Campaign Analytics
       |   (Funnel, Affinity)
       |
       +-- Model Diagnostics
       |   (Threshold, Audit)
       |
       +-- Economic Intelligence
           (Value, Cost, EVaR)
       |
       v
   Streamlit Dashboard
   (9 Functional Views)
```
 
---
 
## Data Schema
 
### Raw Input Files
 
**Customers**
```
customer_id | gender | age | city | registration_date
```
 
**Transactions**
```
transaction_id | customer_id | transaction_date | bill_amount | outlet
```
 
**Campaigns**
```
campaign_id | customer_id | campaign_date | campaign_type | reward_type
reward_value | campaign_cost | sent | delivered | clicked | redeemed
redemption_date
```
 
### Validation Controls
 
Uploaded data is checked for:
- Required and unexpected columns
- Data type correctness
- Date validity
- Missing values and duplicates
- Identifier validity
- Customer reference integrity
- Cross-file consistency
- File size limits (50 MB max)
- Row limits (500,000 per file)
- Upload count limits (4 files per upload)
---
 
## Feature Engineering
 
### RFM Metrics
 
**Recency** - Days since the customer's last purchase
 
**Frequency** - Total number of purchases made
 
**Monetary** - Total customer spending
 
 
### Customer Features
 
| Feature | Description |
|---------|-------------|
| Customer Tenure | Days since registration |
| Average Bill | Mean transaction value |
| Average Purchase Gap | Average days between purchases |
| Campaigns Sent | Number of campaigns delivered to customer |
| Campaigns Clicked | Number of campaigns clicked |
| Campaigns Redeemed | Number of campaigns redeemed |
| Delivery Rate | Percentage of campaigns delivered |
| Click Rate | Percentage of campaigns clicked |
| Redemption Rate | Percentage of campaigns redeemed |
 
 
### Churn Definition
 
```
churn = 1 → No purchase during the next 90 days
churn = 0 → At least one purchase during the next 90 days
```
 
The model estimates the probability that a customer will fail to make a purchase during the following 90 days, given information available at the observation date.
 
---
 
## Machine Learning Models
 
### Models Evaluated
 
Three supervised models were trained and evaluated:
 
**Logistic Regression**
- Accuracy: 77.87%
- Precision: 42.51%
- Recall: 68.16%
- F1-Score: 52.36%
- ROC-AUC: 0.8172
**Decision Tree**
- Accuracy: 83.55%
- Precision: 53.72%
- Recall: 56.42%
- F1-Score: 55.04%
- ROC-AUC: 0.7293
**Random Forest (Selected)**
- Accuracy: 89.33%
- Precision: 86.00%
- Recall: 48.04%
- F1-Score: 61.65%
- ROC-AUC: 0.8639
### Why Random Forest?
 
Random Forest was selected based on the highest ROC-AUC score (0.8639) among all evaluated models.
 
### Confusion Matrix (Random Forest)
 
```
True Negatives:  810
False Positives:  14
False Negatives:  93
True Positives:   86
```
 
---
 
## Risk Classification
 
### Churn Risk Levels
 
Predicted churn probabilities are converted into four business-facing risk levels:
 
| Probability | Risk Level |
|------------|-----------|
| < 0.25 | Low |
| 0.25 – 0.50 | Medium |
| 0.50 – 0.75 | High |
| >= 0.75 | Very High |
 
These are operational classification rules, not guarantees of customer behavior.
 
---
 
## Customer Segmentation
 
### RFM Segments
 
The system classifies customers into seven behavioral segments based on Recency, Frequency, and Monetary value:
 
**Champions**
Customers with excellent engagement and high value. Best retention targets.
 
**Loyal Customers**
Consistent, valuable customers with regular purchase behavior.
 
**Potential Loyalists**
High engagement level but lower monetary value. Growth opportunity segment.
 
**Regular Customers**
Steady, moderate-value customers with consistent behavior.
 
**At Risk**
Customers showing declining engagement and decreasing purchase frequency.
 
**Inactive**
Customers with no recent purchase activity.
 
**Lost**
Customers with very limited historical engagement or activity.
 
Note: RFM segments describe historical behavior only. They do not replace the predictive churn model.
 
---
 
## Retention Priority Scoring
 
### Priority Score Calculation
 
Retention Priority combines predicted churn risk and customer value:
 
```
Retention Priority = (60% × Churn Probability) + (40% × Customer Value Percentile)
```
 
Score ranges from 0 to 100.
 
| Score | Priority Level |
|-------|-----------------|
| < 25 | Low |
| 25 – 50 | Medium |
| 50 – 75 | High |
| >= 75 | Critical |
 
### Expected Revenue at Risk (EVaR)
 
```
EVaR = Churn Probability × Historical Customer Value
```
 
**Example:**
- Historical customer value: Rs. 20,000
- Predicted churn probability: 70%
- Expected Value at Risk: Rs. 14,000
EVaR represents probability-weighted economic exposure. It does not represent guaranteed revenue loss.
 
---
 
## Campaign Performance
 
### Funnel Metrics
 
Current analytical dataset performance:
 
| Stage | Count | Rate |
|-------|-------|------|
| Sent | 8,000 | 100% |
| Delivered | 7,311 | 91.39% |
| Clicked | 2,236 | 30.58% |
| Redeemed | 701 | 9.59% |
 
 
### Campaign Analytics
 
The system tracks the complete campaign funnel:
 
**Sent** → Customers who received a campaign offer
 
**Delivered** → Campaigns successfully delivered
 
**Clicked** → Customers who clicked on the campaign
 
**Redeemed** → Customers who completed the campaign redemption
 
Campaign performance can be examined by campaign type and customer segment.
 
### Campaign Affinity
 
Campaign Affinity analyzes observed customer response patterns across different campaign types.
 
Current analysis includes:
- 4,421 customer-campaign type interactions
- 999 unique customers
- 6 different campaign types
- 7,311 delivered exposures
- 2,236 total clicks
Note: Campaign Affinity is descriptive analysis. It does not establish causal campaign effectiveness or individual customer preferences.
 
---
 
## Economic Intelligence
 
### Customer Value
 
Customer Value represents the historical monetary contribution based on transaction activity.
 
Includes:
- Total historical revenue
- Purchase count
- Average order value
- Active days
- Annualized revenue estimates
This is a historical metric, not a predictive lifetime-value model.
 
### Retention Cost
 
Retention Cost quantifies historical campaign expenditure.
 
Supports:
- Campaign-level retention cost
- Customer-level retention cost
- Total and average retention costs
- Analysis by campaign type
Historical campaign cost is an observed cost measure. It is not automatically interpreted as the cost of a future intervention.
 
### Net Value at Risk
 
```
Net Value at Risk = Expected Value at Risk - Retention Cost
```
 
This provides a cost-adjusted economic exposure measure.
 
### Economic Priority
 
Economic Priority ranks customers using Net Value at Risk into four categories:
 
| Level | Description |
|-------|-------------|
| Low | Low economic exposure |
| Medium | Moderate exposure requiring attention |
| High | Significant exposure |
| Critical | Highest priority for retention action |
 
---
 
## Streamlit Dashboard
 
The application includes 9 functional views plus a home page.
 
### Data & Upload
 
- CSV upload workflow
- Schema validation
- Data quality reporting
- Active dataset management
### Customer Search
 
- Customer profile lookup
- Behavior exploration
- Transaction history
- Campaign history
### Model Diagnostics
 
- Churn threshold sensitivity analysis
- Precision/Recall/F1 across different thresholds
- Segment-level model performance
- Model calibration assessment
### Data Quality
 
- Dataset validation results
- Missing value analysis
- Duplicate detection
- Cross-file consistency checks
### Customer Intelligence
 
- Customer behavior analysis
- Segmentation overview
- Engagement metrics
- RFM distributions
### Retention & Model Insights
 
- Churn risk distribution
- Retention priority ranking
- Feature importance
- Customer risk profiles
### Campaign Performance
 
- Campaign funnel visualization
- Channel performance
- Campaign-type analysis
- Delivery rates and engagement
### Campaign Affinity
 
- Customer response by campaign type
- Click and redemption rates
- Segment affinity analysis
- Campaign type effectiveness
### Economic Intelligence
 
- Customer value distribution
- Retention cost analysis
- Expected Value at Risk (EVaR)
- Economic priority ranking
- Cost-adjusted prioritization
---
 
## Visual Design
 
The dashboard uses a unified visual system:
 
- Dark-blue application shell
- Cream-colored information cards
- Warm brown accents
- Consistent KPI and metric cards
- Plotly-based analytical charts
- Responsive layouts for different screen sizes
---
 
## Project Structure
 
```
Customer-Churn-Prediction/
 
.streamlit/
└── config.toml
 
app/
├── app.py                          (Main Streamlit application)
├── campaign_affinity_page.py
├── customer_search_page.py
├── data_quality_page.py
├── data_upload_page.py
├── economic_pages.py
└── model_diagnostics_page.py
 
data/
├── analytics/
│   ├── campaign_outcomes.csv
│   └── synthetic_campaign_experiment.csv
└── training/
    ├── campaigns.csv
    ├── customer_features.csv
    ├── customers.csv
    └── transactions.csv
 
models/
├── decision_tree.joblib
├── logistic_regression.joblib
└── random_forest.joblib
 
src/
├── analytics/
│   ├── campaign.py
│   ├── campaign_affinity.py
│   ├── churn_threshold.py
│   ├── customer.py
│   ├── customer_search.py
│   ├── customer_value.py
│   ├── custom_model_evaluation.py
│   ├── data_quality.py
│   ├── economic_priority.py
│   ├── economic_value.py
│   ├── failure_modes.py
│   ├── model_calibration.py
│   ├── retention.py
│   ├── retention_cost.py
│   ├── segment_model_audit.py
│   └── segmentation.py
│
├── data/
│   ├── analytics_loader.py
│   ├── custom_pipeline.py
│   ├── loader.py
│   ├── schema.py
│   ├── templates.py
│   ├── upload.py
│   └── validator.py
│
├── features/
│   ├── churn_label.py
│   └── feature_engineering.py
│
├── ml/
│   ├── evaluate.py
│   ├── feature_importance.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   ├── risk.py
│   └── train.py
│
├── preprocessing/
│   └── preprocessing.py
│
└── visualization/
    └── charts.py
 
tests/
└── (Comprehensive pytest suite)
 
.gitignore
config.py
requirements.txt
README.md
```
 
---
 
## Testing
 
The repository includes a comprehensive automated test suite covering all major components:
 
**Data Pipeline**
- Schema validation
- Data quality checks
- Upload validation
- Resource limit enforcement
- Date and type conversion
- Missing value handling
- Duplicate detection
- Customer reference validation
**Feature Engineering**
- RFM feature calculation
- Campaign features
- Customer tenure
- Redemption tracking
- Feature generation
**Machine Learning**
- Model training
- Model evaluation
- Model persistence
- Prediction accuracy
- Risk classification
- Feature importance
**Analytics**
- Customer segmentation
- Retention priority scoring
- Customer value calculation
- Retention cost analysis
- Economic Value at Risk
- Economic priority ranking
- Campaign analytics
- Campaign affinity analysis
**Dashboard**
- Streamlit page behavior
- Visualization rendering
- Data loading and caching
**Total Test Coverage**: 201 passing automated tests
 
Run the complete test suite:
```bash
pytest -q
```
 
---
 
## Installation & Setup
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```
 
### 2. Create a Virtual Environment
 
```bash
python -m venv .venv
```
 
### 3. Activate the Virtual Environment
 
On Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```
 
On macOS/Linux:
```bash
source .venv/bin/activate
```
 
### 4. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 5. Run the Streamlit Application
 
```bash
streamlit run app/app.py
```
 
The application will automatically open in your default browser.
 
---
 
## Analytical Limitations & Disclaimers
 
### Churn Prediction
 
Predicted churn probabilities are model outputs based on historical customer behavior. They are not guarantees of future customer behavior. A customer's predicted churn probability represents what the model learned from past patterns, which may not perfectly reflect real-world outcomes.
 
### Probability Interpretation
 
Predicted probabilities should not automatically be interpreted as perfectly calibrated real-world probabilities. Treat them as relative risk rankings.
 
### Feature Importance
 
Random Forest feature importance describes the contribution of features to model behavior. It does not establish causal drivers of churn. Important features are correlated with churn, not necessarily causal.
 
### RFM Segmentation
 
RFM segments describe historical customer behavior patterns. They do not represent immutable customer identities or guaranteed future outcomes. A customer may move between segments as behavior changes.
 
### Retention Priority
 
Retention Priority is a business scoring framework based on churn probability and relative customer value. It is a prioritization tool for focusing retention resources. It is not a prediction of intervention success or guaranteed retention outcome.
 
### Expected Value at Risk (EVaR)
 
EVaR represents probability-weighted economic exposure. It does not represent:
- Guaranteed revenue loss
- Guaranteed savings from intervention
- Guaranteed recovery of at-risk revenue
### Retention Cost
 
Historical campaign cost is an observed cost measure from past campaigns. It should not automatically be interpreted as the cost of a future retention intervention, which may differ substantially.
 
### Campaign Analysis
 
Observed campaign response patterns are descriptive. They do not establish:
- Causal campaign effectiveness
- Individual customer preferences
- Guaranteed future response rates
- Incremental revenue from specific campaigns
Synthetic campaign experiment data is generated analytical data. It must not be presented as real customer evidence or real-world causal evidence.
 
### Intervention Impact
 
This project does not estimate:
- Treatment effects
- Uplift from retention actions
- Guaranteed incremental revenue
- ROI of specific interventions
---
 
## Documentation
 
Detailed technical documentation is available in:
 
`docs/PROJECT_DOCUMENTATION.docx` - Contains implementation history, methodology, testing evidence, and technical notes.
 
---
 
## GitHub Repository
 
Repository: https://github.com/Ayush-Sarkar-04/Customer-Churn-Prediction
 
---
 
## Project Summary
 
Customer Churn Prediction & Campaign Analytics System demonstrates how a churn model can be extended beyond simple prediction into a comprehensive decision-support system.
 
```
Customer Data
+ Transaction Behaviour
+ Campaign Activity
+ Machine Learning
+ RFM Segmentation
+ Retention Prioritization
+ Campaign Analytics
+ Economic Intelligence
= Holistic Customer Intelligence Platform
```
 
The system provides decision-makers with multiple analytical lenses to understand customer behavior, assess retention risk, prioritize resources, and evaluate campaign effectiveness.
 
---

