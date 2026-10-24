
import pandas as pd
from pathlib import Path

data_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(data_path)

print("\n========== CUSTOMER CHURN DATASET ==========")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null values:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# Data Cleaning
print("\n========== DATA CLEANING ==========")

print("\nDuplicate rows:", df.duplicated().sum())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print(
    "\nMissing TotalCharges after conversion:",
    df["TotalCharges"].isna().sum()
)

df = df.dropna(subset=["TotalCharges"]).copy()

print("\nDataset shape after cleaning:", df.shape)
print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)


# Exploratory Data Analysis (EDA)

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Overall churn rate
churn_rate = (df["Churn"] == "Yes").mean() * 100

print(f"\nOverall churn rate: {churn_rate:.2f}%")

# 2. Churn by contract type
print("\nChurn rate by contract type:")

contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

print(contract_churn)

# 3. Average tenure of churned and retained customers
print("\nAverage customer tenure:")

tenure_analysis = df.groupby("Churn")["tenure"].mean().round(2)

print(tenure_analysis)

# 4. Average monthly charges
print("\nAverage monthly charges:")

charges_analysis = df.groupby("Churn")["MonthlyCharges"].mean().round(2)

print(charges_analysis)

# 5. Churn by payment method
print("\nChurn rate by payment method:")

payment_churn = (
    df.groupby("PaymentMethod")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .round(2)
)

print(payment_churn.sort_values(ascending=False))


import matplotlib.pyplot as plt
from pathlib import Path

# Create a folder to save charts
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Chart 1: Overall churn distribution
churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])

plt.figure(figsize=(7, 5))
plt.bar(
    ["Stayed", "Churned"],
    churn_counts.values
)
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("reports/figures/churn_distribution.png", dpi=300)
plt.close()

# Chart 2: Churn rate by contract type
contract_rates = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))
plt.bar(contract_rates.index, contract_rates.values)
plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig("reports/figures/churn_by_contract.png", dpi=300)
plt.close()

# Chart 3: Average tenure by churn status
tenure_means = df.groupby("Churn")["tenure"].mean().reindex(["No", "Yes"])

plt.figure(figsize=(7, 5))
plt.bar(["Stayed", "Churned"], tenure_means.values)
plt.title("Average Tenure by Customer Status")
plt.xlabel("Customer Status")
plt.ylabel("Average Tenure (Months)")
plt.tight_layout()
plt.savefig("reports/figures/average_tenure.png", dpi=300)
plt.close()

print("\n========== CHARTS CREATED ==========")
print("reports/figures/churn_distribution.png")
print("reports/figures/churn_by_contract.png")
print("reports/figures/average_tenure.png")


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Prepare data for machine learning
print("\n========== MACHINE LEARNING PREPARATION ==========")

# Remove customerID because it is only an identifier
X = df.drop(columns=["Churn", "customerID"])
y = (df["Churn"] == "Yes").astype(int)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Split the data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])
print("Numerical features:", numeric_features)
print("Categorical features:", categorical_features)
print("Training churn rate:", round(y_train.mean() * 100, 2), "%")
print("Testing churn rate:", round(y_test.mean() * 100, 2), "%")


# Train the first machine-learning model
print("\n========== LOGISTIC REGRESSION MODEL ==========")

# Preprocess numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Create the complete model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train using training data only
model.fit(X_train, y_train)

# Predict churn on unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("\nModel training completed successfully!")

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("\nConfusion Matrix:")
print("Rows = actual, columns = predicted")
print("Order: Stayed, Churned")
print(confusion_matrix(y_test, y_pred))


from sklearn.base import clone

print("\n========== THRESHOLD EXPERIMENT ==========")

# Split training data into a smaller training set and validation set
X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Create a fresh copy of the pipeline
tuning_model = clone(model)
tuning_model.fit(X_subtrain, y_subtrain)

# Probability that each validation customer will churn
valid_probabilities = tuning_model.predict_proba(X_valid)[:, 1]

# Compare several classification thresholds
print("\nThreshold | Precision | Recall | F1-score")
print("-" * 43)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    valid_predictions = (
        valid_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid, valid_predictions, zero_division=0
    )
    recall = recall_score(
        y_valid, valid_predictions, zero_division=0
    )
    f1 = f1_score(
        y_valid, valid_predictions, zero_division=0
    )

    print(
        f"{threshold:>9.2f} | "
        f"{precision:>9.4f} | "
        f"{recall:>6.4f} | "
        f"{f1:>8.4f}"
    )


# Final Logistic Regression evaluation
print("\n========== FINAL MODEL: THRESHOLD 0.30 ==========")

# Train on all training data
model.fit(X_train, y_train)

# Get churn probabilities for unseen test customers
test_probabilities = model.predict_proba(X_test)[:, 1]

# Apply the selected threshold
selected_threshold = 0.30
final_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

# Evaluate final predictions
print(f"\nSelected threshold: {selected_threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, final_predictions):.4f}")
print(
    "Precision: "
    f"{precision_score(y_test, final_predictions, zero_division=0):.4f}"
)
print(
    "Recall:    "
    f"{recall_score(y_test, final_predictions, zero_division=0):.4f}"
)
print(
    "F1-score:  "
    f"{f1_score(y_test, final_predictions, zero_division=0):.4f}"
)
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("\nConfusion Matrix:")
print("Rows = actual, columns = predicted")
print("Order: Stayed, Churned")
print(confusion_matrix(y_test, final_predictions))


from sklearn.ensemble import RandomForestClassifier

print("\n========== RANDOM FOREST MODEL ==========")

# Reuse the same preprocessing pipeline
random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Train on the same training set
random_forest.fit(X_train, y_train)

# Predict on the same untouched test set
rf_predictions = random_forest.predict(X_test)
rf_probabilities = random_forest.predict_proba(X_test)[:, 1]

# Evaluate using the standard 0.50 threshold
print("\nEvaluation at threshold 0.50:")
print(f"Accuracy:  {accuracy_score(y_test, rf_predictions):.4f}")
print(
    "Precision: "
    f"{precision_score(y_test, rf_predictions, zero_division=0):.4f}"
)
print(
    "Recall:    "
    f"{recall_score(y_test, rf_predictions, zero_division=0):.4f}"
)
print(
    "F1-score:  "
    f"{f1_score(y_test, rf_predictions, zero_division=0):.4f}"
)
print(f"ROC-AUC:   {roc_auc_score(y_test, rf_probabilities):.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))


from sklearn.base import clone

print("\n========== RANDOM FOREST THRESHOLD EXPERIMENT ==========")

# Train on the sub-training set, not the validation or test set
rf_tuning_model = clone(random_forest)
rf_tuning_model.fit(X_subtrain, y_subtrain)

# Get churn probabilities for validation customers
rf_valid_probabilities = rf_tuning_model.predict_proba(X_valid)[:, 1]

print("\nThreshold | Precision | Recall | F1-score")
print("-" * 43)

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    rf_valid_predictions = (
        rf_valid_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_valid, rf_valid_predictions, zero_division=0
    )
    recall = recall_score(
        y_valid, rf_valid_predictions, zero_division=0
    )
    f1 = f1_score(
        y_valid, rf_valid_predictions, zero_division=0
    )

    print(
        f"{threshold:>9.2f} | "
        f"{precision:>9.4f} | "
        f"{recall:>6.4f} | "
        f"{f1:>8.4f}"
    )


print("\n========== FINAL RANDOM FOREST: THRESHOLD 0.40 ==========")

# Retrain on the full training set
random_forest.fit(X_train, y_train)

# Predict churn probabilities on the untouched test set
rf_test_probabilities = random_forest.predict_proba(X_test)[:, 1]

# Apply the threshold selected using validation data
rf_selected_threshold = 0.40
rf_final_predictions = (
    rf_test_probabilities >= rf_selected_threshold
).astype(int)

# Calculate final test metrics
rf_final_accuracy = accuracy_score(y_test, rf_final_predictions)
rf_final_precision = precision_score(
    y_test, rf_final_predictions, zero_division=0
)
rf_final_recall = recall_score(
    y_test, rf_final_predictions, zero_division=0
)
rf_final_f1 = f1_score(
    y_test, rf_final_predictions, zero_division=0
)
rf_final_auc = roc_auc_score(y_test, rf_test_probabilities)

print(f"\nSelected threshold: {rf_selected_threshold:.2f}")
print(f"Accuracy:  {rf_final_accuracy:.4f}")
print(f"Precision: {rf_final_precision:.4f}")
print(f"Recall:    {rf_final_recall:.4f}")
print(f"F1-score:  {rf_final_f1:.4f}")
print(f"ROC-AUC:   {rf_final_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    rf_final_predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("\nConfusion Matrix:")
print("Rows = actual, columns = predicted")
print("Order: Stayed, Churned")
print(confusion_matrix(y_test, rf_final_predictions))


# ================= MODEL COMPARISON =================

# Logistic Regression test probabilities and predictions
lr_test_probabilities = model.predict_proba(X_test)[:, 1]
lr_default_predictions = (
    lr_test_probabilities >= 0.50
).astype(int)

lr_selected_threshold = 0.30
lr_selected_predictions = (
    lr_test_probabilities >= lr_selected_threshold
).astype(int)

# Random Forest test probabilities and predictions
rf_default_predictions = (
    rf_test_probabilities >= 0.50
).astype(int)

rf_selected_predictions = (
    rf_test_probabilities >= rf_selected_threshold
).astype(int)


def calculate_metrics(model_name, threshold, y_true, predictions, probabilities):
    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, probabilities)
    }


comparison = pd.DataFrame([
    calculate_metrics(
        "Logistic Regression", 0.50, y_test,
        lr_default_predictions, lr_test_probabilities
    ),
    calculate_metrics(
        "Logistic Regression", lr_selected_threshold, y_test,
        lr_selected_predictions, lr_test_probabilities
    ),
    calculate_metrics(
        "Random Forest", 0.50, y_test,
        rf_default_predictions, rf_test_probabilities
    ),
    calculate_metrics(
        "Random Forest", rf_selected_threshold, y_test,
        rf_selected_predictions, rf_test_probabilities
    )
])

output_dir = Path("reports/tables")
output_dir.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output_dir / "model_comparison.csv",
    index=False,
    float_format="%.4f"
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.round(4).to_string(index=False))
print("\nSaved model comparison to reports/tables/model_comparison.csv")


# ================= MODEL PERFORMANCE CHART =================

import matplotlib.pyplot as plt

figures_dir = Path("reports/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

chart_data = comparison.copy()
chart_data["Model and Threshold"] = (
    chart_data["Model"]
    + " ("
    + chart_data["Threshold"].map(lambda x: f"{x:.2f}")
    + ")"
)

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]

ax = chart_data.set_index("Model and Threshold")[metrics_to_plot].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1),
    rot=15
)

ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model (decision threshold)")
ax.legend(title="Metric")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    figures_dir / "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved updated chart to reports/figures/model_performance_comparison.png")



