import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Medical Insurance EDA Dashboard",
    page_icon="🏥",
    layout="wide"
)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.autolayout"] = True

# ================= LOAD DATA =================
df = pd.read_csv("insurance.csv")

# ================= FEATURE GROUPS =================
num_features = ["age", "bmi", "children", "charges"]
cat_features = ["sex", "smoker", "region"]

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.kpi-box {
    border: 5px solid #e0e0e0;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    background-color: #ffffff;
}
.kpi-title {
    font-size: 14px;
    color: #6c757d;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>🏥 Medical Insurance EDA Dashboard</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

# ================= KPI SECTION =================
total_records = df.shape[0]
avg_age = round(df["age"].mean(), 1)
avg_bmi = round(df["bmi"].mean(), 1)
avg_charges = round(df["charges"].mean(), 2)
smoker_percent = round((df["smoker"].value_counts(normalize=True)["yes"]) * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(f"""
<div class="kpi-box">
<div class="kpi-title">Total Records</div>
<div class="kpi-value">{total_records}</div>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi-box">
<div class="kpi-title">Avg Age</div>
<div class="kpi-value">{avg_age}</div>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi-box">
<div class="kpi-title">Avg BMI</div>
<div class="kpi-value">{avg_bmi}</div>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi-box">
<div class="kpi-title">Avg Charges</div>
<div class="kpi-value">{avg_charges}</div>
</div>
""", unsafe_allow_html=True)

k5.markdown(f"""
<div class="kpi-box">
<div class="kpi-title">Smokers (%)</div>
<div class="kpi-value">{smoker_percent}%</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.header("🛠️ Analysis Controls")

analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Univariate Analysis", "Bivariate Analysis"]
)

# ================= UNIVARIATE ANALYSIS =================
if analysis_type == "Univariate Analysis":

    st.subheader("📊 Univariate Analysis")

    feature = st.sidebar.selectbox(
        "Select Feature",
        num_features + cat_features
    )

    col1, col2 = st.columns(2)

    if feature in num_features:
        with col1:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df[feature], bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x=df[feature], ax=ax)
            ax.set_title(f"Boxplot of {feature}")
            st.pyplot(fig)

    else:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x=df[feature], ax=ax)
        ax.set_title(f"Count of {feature}")
        st.pyplot(fig)

# ================= BIVARIATE ANALYSIS =================
else:

    st.subheader("📈 Bivariate Analysis")

    bi_type = st.sidebar.selectbox(
        "Select Relationship",
        ["Num vs Num", "Num vs Cat", "Cat vs Cat"]
    )

    if bi_type == "Num vs Num":
        x = st.sidebar.selectbox("X Axis", num_features)
        y = st.sidebar.selectbox("Y Axis", num_features, index=1)

        fig, ax = plt.subplots(figsize=(7,4))
        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue="smoker",
            palette="viridis",
            alpha=0.6,
            ax=ax
        )
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

    elif bi_type == "Num vs Cat":
        num = st.sidebar.selectbox("Numerical Feature", num_features)
        cat = st.sidebar.selectbox("Categorical Feature", cat_features)

        fig, ax = plt.subplots(figsize=(7,4))
        sns.boxplot(data=df, x=cat, y=num, ax=ax)
        ax.set_title(f"{num} by {cat}")
        st.pyplot(fig)

    else:
        x = st.sidebar.selectbox("Category", cat_features)
        y = st.sidebar.selectbox("Hue", cat_features, index=1)

        fig, ax = plt.subplots(figsize=(7,4))
        sns.countplot(data=df, x=x, hue=y, ax=ax)
        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

# ================= FOOTER =================
st.markdown("---")
st.info(
    "This dashboard performs Exploratory Data Analysis (EDA) on the Medical Insurance dataset "
    "using interactive visualizations."
)
