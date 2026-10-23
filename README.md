# Customer Churn Prediction

A machine learning project that analyzes customer churn patterns and compares classification models to identify customers who may be likely to leave a telecommunications service.

## Project Overview

Customer churn is an important business problem because losing existing customers can affect revenue and long-term customer relationships.

This project uses the IBM Telco Customer Churn dataset to explore customer behavior, identify patterns associated with churn, and evaluate machine learning models for predicting whether a customer will leave.

## Objectives

- Explore customer demographics, service subscriptions, contracts, and billing information.
- Clean and prepare the dataset for analysis.
- Investigate churn patterns using exploratory data analysis (EDA).
- Build and compare Logistic Regression and Random Forest classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Investigate how classification thresholds affect churn detection.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- VS Code

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The original dataset contains 7,043 customer records and 21 columns. During preprocessing, 11 records with invalid or missing `TotalCharges` values were removed, leaving 7,032 records for analysis.

The target variable is `Churn`:

- `Yes`: The customer churned.
- `No`: The customer stayed.

The dataset includes customer tenure, contract type, monthly charges, payment method, internet services, and other customer characteristics.

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── WA_Fn-Use-C_-Telco-Customer-Churn.csv
├── reports/
│   ├── figures/
│   │   ├── average_tenure.png
│   │   ├── churn_by_contract.png
│   │   ├── churn_distribution.png
│   │   └── model_performance_comparison.png
│   └── tables/
│       └── model_comparison.csv
├── churn_analysis.py
└── README.md
```

## Data Preparation

The preprocessing steps include:

- Converting `TotalCharges` to a numeric data type.
- Removing records with invalid `TotalCharges` values.
- Checking for duplicate records.
- Separating the features from the target variable.
- Excluding the customer ID from model features.
- Splitting the data into training and testing sets using stratified sampling.
- Imputing missing values, scaling numerical features, and one-hot encoding categorical features through Scikit-learn pipelines.

The training set contains 5,625 records, and the test set contains 1,407 records.

## Exploratory Data Analysis

The analysis identified several patterns:

- **Contract type:** Month-to-month customers had a churn rate of approximately 42.71%, compared with 11.28% for one-year contracts and 2.85% for two-year contracts.
- **Customer tenure:** Customers who churned had an average tenure of approximately 18 months, compared with approximately 38 months for customers who stayed.
- **Monthly charges:** Customers who churned had higher average monthly charges (approximately 74.44) than customers who stayed (approximately 61.31).
- **Payment method:** Customers using electronic checks had a churn rate of approximately 45.29%, the highest among the payment methods examined.

These are associations in the dataset and do not establish that any particular factor causes churn.

### Visualizations

**Churn Distribution**

![Churn Distribution](reports/figures/churn_distribution.png)

**Churn by Contract Type**

![Churn by Contract Type](reports/figures/churn_by_contract.png)

**Average Customer Tenure**

![Average Tenure](reports/figures/average_tenure.png)

**Model Performance Comparison**

![Model Performance Comparison](reports/figures/model_performance_comparison.png)

## Machine Learning Models

Two classification algorithms were evaluated:

### 1. Logistic Regression

A baseline classification model used to estimate the probability of customer churn.

### 2. Random Forest

An ensemble learning model configured with 300 trees, a maximum depth of 12, a minimum leaf size of 3, and balanced class weights.

Different classification thresholds were also investigated to understand the trade-off between precision and recall.

## Model Evaluation

The following table summarizes the recorded test-set results from the four experiments.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 80.38% | 64.85% | 57.22% | 60.80% | 83.59% |
| Logistic Regression | 0.30 | 74.27% | 51.08% | 75.94% | 61.08% | 83.59% |
| Random Forest | 0.50 | 76.97% | 55.12% | 71.93% | 62.41% | 83.07% |
| Random Forest | 0.40 | 73.21% | 49.75% | 81.02% | 61.65% | 83.07% |

The Random Forest model at a threshold of 0.50 achieved the highest F1-score among these experiments. The Random Forest model at 0.40 achieved the highest recall, detecting 303 of the 374 customers who churned in the test set.

Lowering the threshold increases the number of customers flagged as potential churners. This can help identify more customers at risk, but it also increases false positives.

**Evaluation note:** These figures are exploratory comparisons. Because multiple configurations were evaluated on the same test set, their results should not be treated as a final independent model evaluation. A stricter final evaluation would select the model and threshold using training/validation data and evaluate the selected configuration on a previously untouched test set.

## How to Run the Project

### Prerequisites

- Python installed
- Visual Studio Code or another Python-compatible editor

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install pandas matplotlib scikit-learn
```

### 4. Run the analysis

Ensure the dataset is located in the `data/` folder with the filename expected by the script, then run:

```powershell
python churn_analysis.py
```

The script performs data preparation, exploratory analysis, model training and evaluation, and saves figures and comparison results under `reports/`.

## Future Improvements

- Evaluate additional classification algorithms.
- Use cross-validation and validation-based threshold selection.
- Perform a final evaluation on an untouched test set.
- Investigate feature importance and model interpretability.
- Explore explainability techniques to understand individual predictions.
- Develop a simple web application to demonstrate churn predictions.

## Disclaimer

This project is intended for educational and portfolio purposes. Model predictions are estimates and should not be treated as certain outcomes for individual customers.
