# My-Telco-Churn-Prediction

📊 Telco Customer Churn Prediction
🔹 Project Overview

Customer churn is a major challenge for telecom companies. Acquiring new customers is far more expensive than retaining existing ones. This project uses machine learning models to predict whether a customer is likely to churn based on their demographics, account information, and service usage.

The goal is to:

Identify high-risk customers before they leave.

Provide business insights into key churn drivers.

Support customer retention strategies with actionable predictions.

🔹 Dataset

Source: Telco Customer Churn Dataset (Kaggle)

Rows: ~7,043 customers

Columns: 21 (demographics, services, account info, charges)

Target: Churn (Yes = customer left, No = customer stayed)

🔹 Features

Customer Info: Gender, Senior Citizen, Partner, Dependents

Services: Phone, Internet, Online Security, Streaming TV

Account: Contract type, Paperless billing, Payment method

Charges: MonthlyCharges, TotalCharges

Tenure: Months with the company

🔹 Models Implemented

Logistic Regression (baseline model)

Random Forest

XGBoost (best performance)

Metrics evaluated:

Accuracy

Precision, Recall, F1-score

ROC-AUC

🔹 Tools & Libraries

Python (Pandas, NumPy, Scikit-learn, XGBoost)

Visualization: Matplotlib, Seaborn, Plotly

Dashboard: Power BI / Streamlit

🔹 Project Workflow

Data Cleaning & Preprocessing

Handle missing values in TotalCharges

Encode categorical features

Scale numerical features

Exploratory Data Analysis (EDA)

Visualize churn distribution

Correlation between features and churn

Model Training & Evaluation

Logistic Regression baseline

Advanced ML (Random Forest, XGBoost)

Compare metrics

Results Visualization

Feature importance (SHAP values, XGBoost importance)

Power BI dashboard with churn insights

Deployment (Optional)

Streamlit app for interactive churn prediction

🔹 Results

Logistic Regression: Good baseline

XGBoost: Best performing model (higher recall and ROC-AUC)

Key churn factors: Contract type, Tenure, Monthly Charges, Internet services

🔹 Future Improvements

Add deep learning models (Neural Networks)

Improve handling of class imbalance (SMOTE, cost-sensitive learning)

Deploy interactive churn dashboard

🔹 How to Run
# Clone the repository
git clone https://github.com/DonaldCaffey2531/My-Telco-Churn-Prediction
cd telco-customer-churn

# Install dependencies
pip install -r requirements.txt

# Run model training
python churn_model.py

# Run Streamlit app (optional)
streamlit run app.py

🔹 License

This project is licensed under the MIT License.
