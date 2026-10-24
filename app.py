from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f4f6ff;
    color: #172554;
}

.stApp h1,
.stApp h2,
.stApp h3 {
    color: #17336b !important;
}

.stApp p,
.stApp li,
.stApp label {
    color: #263653 !important;
}

[data-testid="stSidebar"] {
    background: #e8ecff;
    border-right: 1px solid #d5dcfa;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #172554 !important;
}

[data-testid="stSelectbox"] label {
    color: #172554 !important;
}

[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] input {
    color: #172554 !important;
}

.hero {
    padding: 12px 0 20px 0;
}

.hero h1 {
    color: #18356b !important;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 4px;
}

.hero p {
    color: #52658b !important;
    font-size: 1rem;
}

.kpi {
    background: #ffffff;
    border: 1px solid #dce3f7;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(55,75,130,0.07);
    min-height: 125px;
}

.kpi-label {
    color: #475878 !important;
    font-size: 0.9rem;
}

.kpi-value {
    color: #18356b !important;
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    color: #18356b !important;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 8px 0;
}

.insight {
    background: #eeefff;
    border: 1px solid #d9ddfb;
    border-radius: 13px;
    padding: 16px;
    color: #263b68 !important;
    margin-bottom: 10px;
}

.insight b {
    color: #17336b !important;
}

[data-testid="stCaptionContainer"] p {
    color: #52617d !important;
}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"] {
    background: #ffffff;
    border-radius: 12px;
}
</style>

    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna(subset=["TotalCharges"]).copy()
    df["Churn"] = df["Churn"].astype(str).str.strip()
    return df


st.sidebar.markdown("## 📊 Churn Analytics")
st.sidebar.caption("Customer insights • Retention intelligence")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Check that the CSV file is inside your project's data folder.")
    st.stop()

df = load_data(DATA_PATH)

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

contract_options = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
payment_options = ["All"] + sorted(
    df["PaymentMethod"].dropna().unique().tolist()
)

selected_contract = st.sidebar.selectbox(
    "Contract type", contract_options
)
selected_payment = st.sidebar.selectbox(
    "Payment method", payment_options
)
selected_churn = st.sidebar.selectbox(
    "Customer status", ["All", "Churned", "Retained"]
)

filtered = df.copy()

if selected_contract != "All":
    filtered = filtered[filtered["Contract"] == selected_contract]

if selected_payment != "All":
    filtered = filtered[filtered["PaymentMethod"] == selected_payment]

if selected_churn == "Churned":
    filtered = filtered[filtered["Churn"] == "Yes"]
elif selected_churn == "Retained":
    filtered = filtered[filtered["Churn"] == "No"]

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>Turning customer data into insights | Predict · Understand · Retain</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No customers match these filters. Please change your selection.")
    st.stop()

# --------------------------------------------------
# KPI calculations: always based on filtered customers
# --------------------------------------------------
total_customers = len(filtered)
churned = int((filtered["Churn"] == "Yes").sum())
churn_rate = churned / total_customers * 100 if total_customers else 0
revenue = filtered["MonthlyCharges"].sum()

kpis = st.columns(4)
kpi_data = [
    ("Total Customers", f"{total_customers:,}", "👥"),
    ("Churned Customers", f"{churned:,}", "⚠️"),
    ("Churn Rate", f"{churn_rate:.1f}%", "📉"),
    ("Monthly Charges Total", f"${revenue:,.0f}", "💳"),
]

for col, (label, value, emoji) in zip(kpis, kpi_data):
    with col:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">{emoji} {label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")

# --------------------------------------------------
# Charts
# --------------------------------------------------
left, right = st.columns(2)

with left:
    st.markdown(
        '<div class="section-title">Churn by Contract Type</div>',
        unsafe_allow_html=True,
    )

    contract_summary = (
        filtered.groupby("Contract", observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    contract_summary = contract_summary.sort_values(
        "Churn Rate", ascending=False
    )

    st.bar_chart(
        contract_summary.set_index("Contract"),
        y="Churn Rate",
        color="#6288e8",
    )

with right:
    st.markdown(
        '<div class="section-title">Churn by Tenure Group</div>',
        unsafe_allow_html=True,
    )

    tenure_bins = [0, 12, 24, 48, float("inf")]
    tenure_labels = ["0–12 months", "13–24 months", "25–48 months", "49+ months"]

    tenure_df = filtered.copy()
    tenure_df["Tenure Group"] = pd.cut(
        tenure_df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
    )

    tenure_summary = (
        tenure_df.groupby("Tenure Group", observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    st.bar_chart(
        tenure_summary.set_index("Tenure Group"),
        y="Churn Rate",
        color="#8a82e8",
    )

left, right = st.columns(2)

with left:
    st.markdown(
        '<div class="section-title">Churn by Payment Method</div>',
        unsafe_allow_html=True,
    )

    payment_summary = (
        filtered.groupby("PaymentMethod")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    st.bar_chart(payment_summary, color="#56bfa5")

with right:
    st.markdown(
        '<div class="section-title">Customer Risk Overview</div>',
        unsafe_allow_html=True,
    )

    # These are descriptive categories, not ML predictions.
    risk_df = filtered.copy()
    risk_df["Risk Tier"] = pd.cut(
        risk_df["tenure"],
        bins=[-1, 12, 24, float("inf")],
        labels=["High tenure-risk proxy", "Medium tenure-risk proxy", "Lower tenure-risk proxy"],
    )

    risk_summary = risk_df["Risk Tier"].value_counts().reindex(
        ["High tenure-risk proxy", "Medium tenure-risk proxy", "Lower tenure-risk proxy"],
        fill_value=0,
    )

    st.bar_chart(risk_summary, color="#ef9eaf")
    st.caption(
        "Tenure-based descriptive groups only; these are not model-generated risk predictions."
    )

# --------------------------------------------------
# Key insights
# --------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="section-title">💡 Key Insights</div>',
    unsafe_allow_html=True,
)

contract_rates = (
    filtered.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

if not contract_rates.empty:
    highest_contract = contract_rates.idxmax()
    highest_contract_rate = contract_rates.max()

    st.markdown(
        f"""
        <div class="insight">
            <b>Contract insight:</b> {highest_contract} customers have the
            highest observed churn rate ({highest_contract_rate:.1f}%)
            among the contract groups in the current selection.
        </div>
        """,
        unsafe_allow_html=True,
    )

if not tenure_summary.empty:
    highest_tenure = tenure_summary.loc[
        tenure_summary["Churn Rate"].idxmax()
    ]

    st.markdown(
        f"""
        <div class="insight">
            <b>Tenure insight:</b> The {highest_tenure["Tenure Group"]} group
            has the highest observed churn rate
            ({highest_tenure["Churn Rate"]:.1f}%).
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Built with Python, Pandas and Streamlit • Customer Churn Analytics Portfolio Project"
)