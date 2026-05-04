# 📊 Telco Customer Churn — Insights & Recommendations

> **Project:** Telco Customer Churn Prediction  
> **Dataset:** IBM Telco Customer Churn — 7,032 customers × 21 features  
> **Model:** Logistic Regression + Random Forest (Jupyter Notebook)  
> **Tool:** Interactive Streamlit Dashboard

---

## 🔢 Executive Summary

| Metric | Value |
|---|---|
| Total Customers Analyzed | 7,032 |
| Customers Churned | 1,869 |
| Overall Churn Rate | **26.6%** |
| Customers Retained | 5,163 (73.4%) |
| High Risk Customers Flagged | **402** |
| Medium Risk Customers | 1,701 |
| Low Risk Customers | 4,929 |

> Over 1 in 4 customers are leaving. With **402 high-risk customers** identified in real time, the business now has a clear, actionable target list for retention intervention.

---

## 🔍 Key Insights

### 1. Contract Type is the Strongest Churn Driver

Month-to-month customers churn at approximately **~42%**, more than double the rate of one-year or two-year contract holders. Customers with longer commitments are significantly more loyal.

```
Month-to-month  ████████████████████████  ~42%
One year        ████████                  ~11%
Two year        ███                        ~3%
```

**Why it matters:** The vast majority of high-risk customers in the flagged list are on month-to-month contracts with tenures under 7 months. These customers have no switching cost.

---

### 2. Internet Service Type Significantly Impacts Churn

Fiber optic subscribers churn at **~42%** — more than double the rate of DSL users (~19%). Customers with no internet service show the lowest churn.

```
Fiber optic     ████████████████████████  ~42%
DSL             ████████████              ~19%
No Internet     ████                       ~7%
```

**Why it matters:** Fiber optic is typically the premium, higher-revenue tier. Losing these customers is disproportionately costly to revenue.

---

### 3. New Customers Are the Most Vulnerable

Churn rate drops steeply with tenure. Customers in the **0–12 month** bracket churn at the highest rate. Customers who cross the 24-month mark are far more likely to remain long term.

```
Tenure 0–12 mo   ████████████████████████  Highest churn
Tenure 12–24 mo  █████████████             Moderate churn
Tenure 24–48 mo  ██████                    Low churn
Tenure 48–72 mo  ████                      Very low churn
```

**Why it matters:** The first year is the critical retention window. Failing to engage new customers early results in permanent attrition.

---

### 4. High-Risk Customers Share a Clear Profile

From the flagged high-risk customer list (churn probability > 0.83), a consistent pattern emerges:

| Attribute | Common Value |
|---|---|
| Contract Type | Month-to-month |
| Tenure | 1–7 months |
| Monthly Charges | $85–$102 |
| Churn Probability | 0.81 – 0.84+ |

These are **new, high-paying, short-commitment customers** — the most valuable and most at-risk segment simultaneously.

---

### 5. Demographics Show No Gender Bias

Churn rates are nearly identical across male and female customers. Gender is not a significant predictor of churn. Senior citizen customers, however, show a higher tendency to churn than non-seniors.

---

### 6. Billing Behaviour Signals Risk

Customers paying via **electronic check** churn at a noticeably higher rate than those using automatic payment methods (bank transfer, credit card). Manual payment methods correlate with disengagement.

---

## 💡 Recommendations

### 🎯 Recommendation 1 — Launch a Contract Upgrade Campaign

**Target:** All month-to-month customers in their first 6 months  
**Action:** Offer discounted first-year rates or service bundles to incentivize upgrades to annual contracts  
**Expected Impact:** Moving even 10% of high-risk month-to-month customers to one-year contracts could reduce churn by ~400 customers

---

### 🎯 Recommendation 2 — Build a New Customer Onboarding Program

**Target:** Customers with tenure < 12 months  
**Action:** Implement a structured 90-day onboarding journey — welcome calls, usage tips, and loyalty check-ins at Day 7, Day 30, and Day 90  
**Expected Impact:** Improving early engagement reduces the steep churn spike observed in the 0–12 month window

---

### 🎯 Recommendation 3 — Prioritize Fiber Optic Retention

**Target:** Fiber optic subscribers on month-to-month contracts  
**Action:** Investigate service quality complaints, proactively offer tech support, and provide speed/reliability upgrades before dissatisfaction becomes a decision to leave  
**Expected Impact:** Retaining fiber optic customers protects the highest monthly revenue segment

---

### 🎯 Recommendation 4 — Act on the 402 High-Risk Customers Now

**Target:** The 402 flagged high-risk customers (downloadable from the dashboard)  
**Action:** Assign a retention team to reach out within 48 hours with personalised offers — contract discounts, service add-ons, or billing adjustments  
**Expected Impact:** Even a 30% success rate converts ~120 customers, protecting significant recurring revenue

---

### 🎯 Recommendation 5 — Incentivise Autopay Adoption

**Target:** Customers paying via electronic check or mailed check  
**Action:** Offer a small monthly discount ($5–$10) for switching to automatic bank transfer or credit card payment  
**Expected Impact:** Autopay reduces friction, increases perceived commitment, and correlates with lower churn across the dataset

---

### 🎯 Recommendation 6 — Senior Customer Retention Program

**Target:** Senior citizen customers (SeniorCitizen = 1)  
**Action:** Offer simplified plans, priority support lines, and dedicated account managers to address the higher churn tendency in this demographic  
**Expected Impact:** Improves retention in an often-overlooked but loyal-when-engaged segment

---

## 📸 Dashboard Previews

| Page | Screenshot |
|------|-----------|
| **Home — KPIs & Quick Stats** | ![Home](data%20set/visuals/Screenshot%202026-05-04%20143629.png) |
| **EDA — Churn Distribution** | ![Churn Dist](data%20set/visuals/Screenshot%202026-05-04%20143654.png) |
| **Prediction — Churn Probability Gauge** | ![Prediction](data%20set/visuals/Screenshot%202026-05-04%20143708.png) |
| **High Risk Customers — Risk Distribution** | ![High Risk](data%20set/visuals/Screenshot%202026-05-04%20143721.png) |
| **High Risk Customers — Exportable List** | ![List](data%20set/visuals/Screenshot%202026-05-04%20143729.png) |

---

## 🧠 Model Details

| Detail | Value |
|---|---|
| Primary Model | Logistic Regression (Streamlit app) |
| Secondary Model | Random Forest Classifier (Notebook) |
| Preprocessing | Label Encoding + Standard Scaling |
| Features Used | 19 customer attributes |
| Risk Thresholds | High > 0.7 · Medium 0.4–0.7 · Low < 0.4 |

---

## 📁 Repository

```
customer-churn-analysis/
├── streamlit_app.py                          # Interactive prediction dashboard
├── churnPrediction.ipynb                     # EDA & model notebook
└── data set/
    ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Source data (7,032 rows)
    ├── high_risk_customers.csv               # Exported high-risk list
    └── visuals/                              # Dashboard screenshots
```

---

## ⚙️ Run the Dashboard

```bash
git clone https://github.com/safvenn/customer-churn-analysis.git
cd customer-churn-analysis
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
cp "data set/WA_Fn-UseC_-Telco-Customer-Churn.csv" .
streamlit run streamlit_app.py
```

---

*Built with Python · Streamlit · scikit-learn · Pandas · Matplotlib · Seaborn*
