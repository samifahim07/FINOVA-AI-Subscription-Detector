# FINOVA AI — Subscription Intelligence Engine

**Machine Learning–Powered Subscription Detection, Recurring Payment Analysis, and Financial Insights**

FINOVA AI's Subscription Intelligence module analyzes banking transaction history to identify potential subscriptions and recurring bills, estimate billing cycles and costs, flag unusual charge increases, and help users review uncertain detections. It combines a LightGBM transaction classifier with historical pattern analysis, a FastAPI service, and an interactive React dashboard.

**Project status:** Working local prototype using synthetic financial data. It is not a production banking integration.

## Features

- **AI-powered transaction classification:** Scores debit transactions for subscription likelihood using a trained LightGBM classifier.
- **Recurring-payment analysis:** Examines merchant-level payment intervals, amount consistency, and historical frequency.
- **Billing frequency estimation:** Estimates monthly, quarterly, or yearly cycles when enough payment history is available.
- **Activity monitoring:** Labels recurring groups as `Likely Active`, `Possibly Inactive`, or `Frequency Unverified` using heuristic rules.
- **Subscription and bill categorization:** Separates digital subscriptions, recurring bills, housing payments, and other candidates.
- **Payment forecasting:** Estimates the next billing date from the last observed charge and inferred billing cycle.
- **Cost intelligence:** Projects monthly and annual recurring costs, grouped by currency and payment category.
- **Charge-change monitoring:** Flags qualifying increases in observed transaction amounts; these do not necessarily indicate subscription plan price changes.
- **Review queue:** Surfaces groups with insufficient evidence rather than presenting them as confirmed subscriptions.
- **User confirmation:** Supports `Confirm`, `Not a subscription`, and undo actions; the local demo stores these decisions separately from model predictions.
- **Responsive dashboard:** React/TypeScript interface with existing FINOVA AI light/dark themes and animated visual elements.

## Architecture

```text
Synthetic Transaction Dataset
            |
            v
Historical Feature Engineering
            |
            v
     LightGBM Classifier
      (threshold: 0.72)
            |
            v
Customer–Account–Merchant Grouping
            |
            v
Recurring Pattern + Category Rules
            |
            v
Billing Estimates, Cost Insights, Alerts
            |
            v
      FastAPI REST Endpoints
            |
            v
   React + TypeScript Dashboard
            |
            v
User Confirmation / Review Feedback
```

**Important:** The classifier predicts individual transactions. Recurring-group detection, category rules, activity flags, and billing forecasts are additional heuristic stages and have not been assigned the classifier's test performance metrics.

## Dataset

Synthetic data was generated using the [Synthetic Banking Dataset Generator](https://github.com/mbazouz/synthetic-bank-dataset).

| Dataset | Rows | Purpose |
|---|---:|---|
| `transactions.csv` | 213,341 | Transaction history, classifier training, historical feature engineering |
| `subscriptions.csv` | 674 | Synthetic subscription reference records used for exploration and schedule analysis |

**Transaction dataset:** 27 original columns, 100 customers, 180 distinct merchants, 7,344 subscription-labeled transactions (3.44%), and 205,997 non-subscription transactions (96.56%).

The deployed inference engine operates on transaction history. It does **not** use `subscriptions.csv` or the ground-truth `is_subscription` label as input when predicting.

### Engineered features

The selected model uses 10 features:

| Feature | Description |
|---|---|
| `amount` | Absolute transaction amount |
| `days_since_last_payment` | Days since the previous payment to the same merchant |
| `prior_merchant_payments` | Number of preceding customer/account/merchant transactions |
| `amount_difference` | Absolute change from the previous payment amount |
| `previous_interval` | Earlier inter-payment interval |
| `interval_consistency` | Difference between consecutive payment intervals |
| `monthly_pattern` | Approximate 27–32 day gap |
| `similar_amount` | Amount difference of 2 or less |
| `quarterly_pattern` | Approximate 85–95 day gap |
| `yearly_pattern` | Approximate 350–380 day gap |

Features are computed from previously observed transactions. Known-label columns such as `is_subscription`, `label`, `category`, and `subcategory` are excluded from model input to reduce shortcut learning and leakage risk. Missing historical intervals are encoded consistently with the training pipeline.

## Model development and evaluation

Five classifiers were trained and compared using a **chronological 70% training / 20% validation / 10% test split**.

| Split | Transactions | Subscription transactions |
|---|---:|---:|
| Training (70%) | 149,338 | 4,247 |
| Validation (20%) | 42,668 | 2,032 |
| Test (10%) | 21,335 | 1,065 |

### Initial validation comparison

Metrics below refer to the **subscription class (1)** using each model's initial decision setting, before final threshold selection.

| Model | Precision | Recall | F1-score | Average precision |
|---|---:|---:|---:|---:|
| Random Forest | 92.56% | 78.94% | 85.21% | 0.9276 |
| XGBoost | 84.84% | 88.44% | 86.60% | 0.9419 |
| **LightGBM** | **88.13%** | **87.70%** | **87.91%** | **0.9446** |
| CatBoost | 79.67% | 90.26% | 84.63% | 0.9437 |
| Extra Trees | 76.37% | 84.15% | 80.07% | 0.8721 |

LightGBM was selected based on validation F1-score and average precision. Subsequent validation-based threshold optimization selected **0.72**. A small hyperparameter search did not outperform the existing LightGBM configuration.

### Final LightGBM test results

| Metric | Test result |
|---|---:|
| **Accuracy** | **99.10%** |
| **Precision (subscription class)** | **92.02%** |
| **Recall (subscription class)** | **89.86%** |
| **F1-score (subscription class)** | **90.93%** |
| Average precision | 0.9547 |
| ROC-AUC | 0.9937 |
| Decision threshold | 0.72 |

Final test confusion matrix:

| Actual / Predicted | Normal | Subscription |
|---|---:|---:|
| Normal | 20,187 | 83 |
| Subscription | 108 | 957 |

**Evaluation scope:** These are results for the transaction classifier on a held-out portion of the same synthetic data source. They are **not** measured end-to-end accuracy for the complete hybrid reporting system, nor evidence of equivalent performance on real banking transactions or unseen banking institutions.

## Technology stack

| Layer | Technologies |
|---|---|
| Machine learning | Python, pandas, NumPy, scikit-learn, LightGBM |
| Model comparison | Random Forest, XGBoost, LightGBM, CatBoost, Extra Trees |
| Model serialization | Python Pickle (`.pkl`) |
| API | FastAPI, Uvicorn |
| Frontend | React, TypeScript, Vite, CSS, Lucide icons |
| Local feedback storage | JSON file (demo only) |

## Project structure

The subscription module is integrated with an existing FINOVA AI financial dashboard:

```text
AI-Personal-Finance/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── dashboard.py
│   │   ├── analytics.py
│   │   ├── subscriptions.py
│   │   └── subscription_engine.py
│   ├── models/
│   │   └── finova_subscription_model.pkl
│   └── data/
│       ├── financial_dashboard_processed.csv
│       └── subscriptions/
│           ├── transactions.csv
│           └── user_feedback.json     # Created by the local feedback API
├── frontend/
│   └── src/
│       ├── App.tsx
│       └── components/
│           ├── SubscriptionIntelligence.tsx
│           └── SubscriptionIntelligence.css
└── notebooks/                       # Experiments and model development
```

The existing `financial_dashboard_processed.csv` powers separate historical dashboard analytics. **Do not combine its totals with the synthetic subscription dataset's totals.** They represent different demonstrations and may use different currencies and dates.

## Running locally

### 1. Install Python dependencies

From the project root on Windows PowerShell:

```powershell
cd backend
python -m pip install fastapi uvicorn pandas numpy scikit-learn lightgbm
```

Use the backend's active virtual environment, if available. Other modules in the full FINOVA AI project may have additional dependencies.

### 2. Configure the local demo

```powershell
$env:FINOVA_SUBSCRIPTIONS_DEMO = "1"
$env:FINOVA_CUSTOMER_ID = "C0000047"
```

Place the **trusted** saved model at `backend/models/finova_subscription_model.pkl` and generated transaction data at `backend/data/subscriptions/transactions.csv`.

### 3. Start FastAPI

From `backend/`:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API documentation: `http://127.0.0.1:8000/docs`

Example report: `http://127.0.0.1:8000/subscriptions/report`

### 4. Start the React frontend

In a separate terminal from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite development URL shown in the terminal (commonly `http://localhost:5173`). Keep the FastAPI server running in a separate terminal.

## Subscription API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/subscriptions/report` | Full customer-specific intelligence report |
| `GET` | `/subscriptions/summary` | Estimated recurring-cost and detection summary |
| `GET` | `/subscriptions/upcoming` | Upcoming estimated payments |
| `GET` | `/subscriptions/review` | Groups requiring further review |
| `GET` | `/subscriptions/feedback` | Locally saved user decisions (when feedback extension is enabled) |
| `POST` | `/subscriptions/feedback` | Save confirmation, rejection, or undo (when feedback extension is enabled) |

### User feedback example

```json
{
  "account_id": "ACCOUNT_ID_FROM_REPORT",
  "merchant_name": "Netflix",
  "currency": "USD",
  "decision": "confirmed"
}
```

Supported decisions are `confirmed`, `rejected`, and `pending` (undo). A decision is associated with a customer/account/merchant/currency group in the local demo. User decisions are persisted independently; they do **not** automatically retrain the classifier or update the model's historical evaluation metrics.

## Example customer-level report

A tested synthetic customer snapshot (`C0000047`) returned:

| Field | Example value |
|---|---:|
| Scanned transactions | 915 |
| ML-positive transaction candidates | 34 |
| Merchant groups | 21 |
| High-confidence recurring groups | 2 |
| Groups needing review | 19 |
| Likely-active groups | 1 |
| Monthly housing estimate | USD 731.74 |
| Annual housing projection | USD 8,780.88 |

These estimates concern the selected synthetic customer's **historical snapshot**, not a live account. In this example, the estimated amount is categorized as **housing**, not confirmed digital subscriptions. A missing digital-subscription cost does not prove that the customer has no subscription services.

## Key limitations

1. **Synthetic-data validation:** Reported model results come from generated financial transactions. External validation on real and unseen customer data is needed.
2. **Class imbalance:** Subscription-labeled transactions represent about 3.44% of the transaction dataset; accuracy alone is not informative.
3. **Group-level inference is heuristic:** A recurring payment is not necessarily a digital subscription. Infrequent or irregular subscriptions can be missed.
4. **Customer-level variation:** Some individual customers can have materially worse results than the overall test-set performance.
5. **Estimated dates and costs:** Forecasts use observed history, not confirmed billing contracts. Variable bills, changed prices, canceled services, and missing records may alter actual payments.
6. **Demo access only:** The current local integration is configured for one demo customer; it does not implement production login, per-user authorization, or real banking connectivity.
7. **Separate feedback storage:** The JSON feedback store is a local prototype, not a multi-user database. Saved decisions do not yet automatically reconcile historical financial forecasts.
8. **Model artifact trust:** Pickle files can execute code during loading. Load only trusted model artifacts.

## Future improvements

- Authenticated multi-user accounts and customer-level access controls.
- Database-backed user confirmations, audit trails, and feedback lifecycle management.
- Merchant normalization and stronger support for new or unseen merchants.
- Evaluation on a new independent dataset, including customer-disjoint holdouts.
- Subscription lifecycle detection, better variable-bill handling, and billing confidence ranges.
- User-verified cost tracking and integration with the broader FINOVA AI Cash-Flow Calendar.
- Monitoring for drift, category errors, false positives, and missed subscriptions.

## Responsible use

This module provides financial analytics and estimated recurring-payment insights. It does not execute payments, cancel subscriptions, connect to live bank accounts, or provide verified account balances. Confirm charges and renewal dates with the underlying service or financial institution before acting on an estimate.

---

**FINOVA AI — Personal Financial Intelligence**  
Subscription Intelligence Module · Local Research and Demonstration Prototype
