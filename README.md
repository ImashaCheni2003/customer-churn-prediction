# Customer Churn Prediction

A machine learning project exploring customer churn in the telecommunications industry using exploratory data analysis, Logistic Regression, and Random Forest.

**Repository:** [GitHub](https://github.com/ImashaCheni2003/customer-churn-prediction)

## Project Overview

Customer churn occurs when customers stop using a company's services. Identifying patterns associated with churn can help businesses better understand customer behaviour and develop retention strategies.

This project analyses the IBM Telco Customer Churn dataset, explores factors associated with customer churn, and compares machine learning classifiers using multiple evaluation metrics.

## Project Objectives

- Clean and prepare customer data for analysis.
- Explore churn patterns across contracts, tenure, billing, and payment methods.
- Build and evaluate classification models.
- Compare model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
- Investigate how classification thresholds affect churn detection.

## Technologies Used

- **Language:** Python
- **Data analysis:** Pandas
- **Visualisation:** Matplotlib
- **Machine learning:** Scikit-learn
- **Development environment:** Visual Studio Code
- **Version control:** Git and GitHub

## Dataset

The project uses the IBM Telco Customer Churn dataset, which originally contains 7,043 customer records and 21 columns.

During preprocessing, 11 records with invalid or missing `TotalCharges` values were removed, leaving **7,032 records** for analysis.

The target variable is `Churn`, which indicates whether a customer left the service (`Yes`) or remained (`No`).

## Key Findings

- **Contract type:** Month-to-month customers had a churn rate of 42.71%, compared with 11.28% for one-year contracts and 2.85% for two-year contracts.
- **Customer tenure:** Customers who churned had an average tenure of approximately 18 months, compared with 38 months for customers who stayed.
- **Monthly charges:** Customers who churned had higher average monthly charges (74.44) than customers who stayed (61.31), in the dataset's recorded monetary units.
- **Payment method:** Electronic-check users had the highest observed churn rate, at 45.29%.

These are observed associations, not evidence that any individual factor directly causes churn.

## Exploratory Data Analysis

### Churn Distribution

![Customer churn distribution](reports/figures/churn_distribution.png)

### Churn by Contract Type

![Churn rate by contract type](reports/figures/churn_by_contract.png)

### Average Customer Tenure

![Average tenure by churn status](reports/figures/average_tenure.png)

## Machine Learning Models

Two classification algorithms were evaluated:

1. **Logistic Regression** — a baseline model for estimating churn probabilities.
2. **Random Forest** — an ensemble classifier configured with 300 trees, a maximum depth of 12, a minimum leaf size of 3, and balanced class weights.

Different classification thresholds were examined to investigate the trade-off between identifying churners and generating false-positive predictions.

## Model Performance

The following results were recorded across four model and threshold experiments.

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 | 80.38% | 64.85% | 57.22% | 60.80% | 83.59% |
| Logistic Regression | 0.30 | 74.27% | 51.08% | 75.94% | 61.08% | 83.59% |
| Random Forest | 0.50 | 76.97% | 55.12% | 71.93% | **62.41%** | 83.07% |
| Random Forest | 0.40 | 73.21% | 49.75% | **81.02%** | 61.65% | 83.07% |

![Model performance comparison](reports/figures/model_performance_comparison.png)

### Interpretation

- Random Forest at a threshold of 0.50 achieved the highest F1-score among the four experiments.
- Random Forest at a threshold of 0.40 achieved the highest recall, identifying 303 of the 374 churners in the test set.
- Lower thresholds can identify more potential churners, but typically increase false positives.
- The preferred model depends on the business objective and the relative costs of missed churners and unnecessary retention interventions.

**Evaluation limitation:** These results are exploratory comparisons because multiple configurations were evaluated on the same test set. They should not be presented as a final independent performance estimate. A stronger final evaluation would select the model and threshold using training/validation data, then evaluate the selected configuration on an untouched test set.

## Project Structure

```text
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── reports/
│   ├── figures/
│   │   ├── average_tenure.png
│   │   ├── churn_by_contract.png
│   │   ├── churn_distribution.png
│   │   └── model_performance_comparison.png
│   └── tables/
│       └── model_comparison.csv
├── churn_analysis.py
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- Git

### 1. Clone the repository

```bash
git clone https://github.com/ImashaCheni2003/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas matplotlib scikit-learn
```

### 4. Run the analysis

```bash
python churn_analysis.py
```

The script runs the data preparation, exploratory analysis, model training, and evaluation workflow. Generated figures and comparison results are saved under `reports/`.

## Future Improvements

- Introduce cross-validation and a dedicated validation set for model selection.
- Perform final evaluation on an untouched test set.
- Investigate feature importance and model explainability.
- Explore additional classification algorithms and systematic hyperparameter tuning.
- Evaluate business-oriented metrics, such as the potential costs of missed churners and false positives.
- Develop an interactive application for demonstrating churn predictions.

## Skills Demonstrated

Data cleaning, exploratory data analysis, data visualisation, feature preprocessing, classification, model evaluation, threshold analysis, Python programming, and version control with Git.

## Disclaimer

This project is intended for educational and portfolio purposes. Model predictions are estimates and should not be interpreted as certain outcomes for individual customers.
