import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Medical Insurance Dashboard & Prediction",
    page_icon="🏥",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

df = load_data()

numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# ================= LOAD MODEL ONCE =================
@st.cache_resource
def load_model():
    import pickle
    return pickle.load(open("insurance_model_new.pkl", "rb"))

# ================= KPI CSS =================
st.markdown("""
<style>
.kpi-card {
    background: linear-gradient(135deg,#1f77b4,#0d47a1);
    padding:18px;
    border-radius:12px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 10px rgba(0,0,0,0.2);
}
.kpi-title{font-size:16px;font-weight:600;}
.kpi-value{font-size:26px;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ================= TAB CSS =================
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {gap:20px;}
.stTabs [data-baseweb="tab"] {
background-color:#f2f2f2;
color:#333 !important;
border-radius:8px 8px 0px 0px;
padding:10px 18px;
font-weight:500;
}
.stTabs [data-baseweb="tab"]:hover {
background-color:#e6e6e6;
color:#000 !important;
}
.stTabs [aria-selected="true"] {
background-color:#ff4b4b !important;
color:white !important;
font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ================= KPI FUNCTION =================
def show_kpis():

    col1,col2,col3,col4,col5 = st.columns(5)

    total = df.shape[0]
    avg_age = round(df["age"].mean(),1)
    avg_bmi = round(df["bmi"].mean(),1)
    avg_charges = round(df["charges"].mean(),2)
    smoker_pct = round(df["smoker"].value_counts(normalize=True)["yes"]*100,1)

    data = [
        ("Total Records",total),
        ("Avg Age",avg_age),
        ("Avg BMI",avg_bmi),
        ("Avg Charges",avg_charges),
        ("Smokers %",smoker_pct)
    ]

    for col,(title,val) in zip([col1,col2,col3,col4,col5],data):
        col.markdown(f"""
        <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{val}</div>
        </div>
        """,unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Dashboard","Model Prediction"])

# ================= DASHBOARD =================
if menu=="Dashboard":

    st.title("🏥 Medical Insurance Dashboard",text_alignment='center')

    tab1,tab2,tab3,tab4 = st.tabs([
        "About Dataset",
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis"
    ])

    # ================= ABOUT DATASET =================
    with tab1:

        show_kpis()
        st.markdown("---")

        st.write("### About This Project")
        st.info("""
        This dashboard analyzes Medical Insurance data to understand how age, BMI,
        smoking habits, region and family size influence medical insurance charges.
        The goal is to uncover cost-driving factors before machine learning modeling.
        """)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("### Dataset Information")
        st.write(f"Shape: {df.shape}")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        st.write("### Project Objective")
        st.info("""
        • Perform Exploratory Data Analysis  
        • Understand healthcare cost drivers  
        • Identify smoker impact on expenses  
        • Prepare features for ML prediction models
        """)

        st.success("""
        **Project Highlights**
        - Smoking dramatically increases charges
        - BMI & Age strongly affect medical expenses
        - Regional differences influence insurance cost
        """)

    # ============ univariate ============================
    # ============ UNIVARIATE ============================
    with tab2:

        show_kpis()
        st.markdown("---")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Histogram","Box Plot","Count Plot"]
        )

        column = st.selectbox("Select Column", df.columns)

        # Histogram
        if chart_type == "Histogram":
            if column in numeric_cols:
                fig = px.histogram(df, x=column, title=f"Histogram of {column}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠ Histogram supports only numeric features.")

        # Box Plot
        elif chart_type == "Box Plot":
            if column in numeric_cols:
                fig = px.box(df, y=column, title=f"Box Plot of {column}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠ Box Plot supports only numeric features.")

        # Count Plot
        elif chart_type == "Count Plot":
            if column in categorical_cols:
                count_df = df[column].value_counts().reset_index()
                count_df.columns = [column,"Count"]

                fig = px.bar(
                    count_df,
                    x=column,
                    y="Count",
                    color=column,
                    title=f"Count Plot of {column}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠ Count Plot supports only categorical features.")
    # ================= bivariate ===========================
    # ================= BIVARIATE ===========================
    with tab3:

        show_kpis()
        st.markdown("---")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Numeric vs Numeric","Numeric vs Categorical","Categorical vs Categorical"]
        )

        graph_type = st.selectbox(
            "Select Graph Type",
            ["Scatter Plot","Box Plot","Bar Chart","Heatmap"]
        )

        # -------- NUMERIC VS NUMERIC --------
        if analysis_type == "Numeric vs Numeric":

            x = st.selectbox("X Feature", numeric_cols)
            y = st.selectbox("Y Feature", numeric_cols, index=1)

            if x == y:
                st.warning("⚠ Please select two different numeric features.")
            elif graph_type == "Scatter Plot":
                fig = px.scatter(df, x=x, y=y, color="smoker")
                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "Heatmap":
                corr = df[[x,y]].corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("⚠ Selected graph is not suitable for Numeric vs Numeric.")

        # -------- NUMERIC VS CATEGORICAL --------
        elif analysis_type == "Numeric vs Categorical":

            num = st.selectbox("Numeric Feature", numeric_cols)
            cat = st.selectbox("Categorical Feature", categorical_cols)

            if graph_type == "Box Plot":

                fig = px.box(
                    df,
                    x=cat,
                    y=num,
                    color=cat,
                    title=f"{num} Distribution across {cat}"
                )

                st.plotly_chart(fig, use_container_width=True)

            elif graph_type == "Bar Chart":

                agg_type = st.selectbox(
                    "Select Aggregation",
                    ["Mean","Median","Sum","Count"]
                )

                if agg_type == "Mean":
                    temp = df.groupby(cat)[num].mean().reset_index()

                elif agg_type == "Median":
                    temp = df.groupby(cat)[num].median().reset_index()

                elif agg_type == "Sum":
                    temp = df.groupby(cat)[num].sum().reset_index()

                else:
                    temp = df.groupby(cat)[num].count().reset_index()

                fig = px.bar(
                    temp,
                    x=cat,
                    y=num,
                    color=cat,
                    title=f"{agg_type} of {num} by {cat}"
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("⚠ Use Box Plot or Bar Chart for Numeric vs Categorical.")
        # -------- CATEGORICAL VS CATEGORICAL --------
        else:

            cat1 = st.selectbox("Category 1", categorical_cols)
            cat2 = st.selectbox("Category 2", categorical_cols, index=1)

            if cat1 == cat2:
                st.warning("⚠ Select different categorical features.")

            else:

                cross = pd.crosstab(df[cat1], df[cat2])

                if graph_type == "Heatmap":
                    fig = px.imshow(cross, text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Bar Chart":

                    cross_reset = cross.reset_index()

                    cross_melt = cross_reset.melt(
                        id_vars=cat1,
                        var_name=cat2,
                        value_name="Count"
                    )

                    fig = px.bar(
                        cross_melt,
                        x=cat1,
                        y="Count",
                        color=cat2,
                        barmode="group",
                        title=f"{cat1} vs {cat2}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("⚠ Use Heatmap or Bar Chart here.")

    # ================= MULTIVARIATE ======================
    with tab4:

        show_kpis()
        st.markdown("---")

        multi_chart = st.selectbox(
            "Select Multivariate Visualization",
            ["Correlation Heatmap","Scatter Matrix","3D Scatter Plot"]
        )

        # Heatmap
        if multi_chart == "Correlation Heatmap":

            corr = df[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Scatter Matrix
        elif multi_chart == "Scatter Matrix":

            fig = px.scatter_matrix(
                df,
                dimensions=numeric_cols,
                color="smoker"
            )
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # 3D Scatter
        else:

            x = st.selectbox("X Axis", numeric_cols)
            y = st.selectbox("Y Axis", numeric_cols, index=1)
            z = st.selectbox("Z Axis", numeric_cols, index=2)

            if len({x,y,z}) < 3:
                st.warning("⚠ Select three different features.")
            else:
                fig = px.scatter_3d(df, x=x, y=y, z=z, color="smoker")
                st.plotly_chart(fig, use_container_width=True)

#============== model prediction ===========================
elif menu == "Model Prediction":

    import numpy as np

    model = load_model()

    # ================= BUTTON CSS =================
    st.markdown("""
    <style>
    div.stButton > button {
        background-color:#1f77b4;
        color:white;
        font-weight:600;
        border-radius:8px;
        padding:10px 18px;
        border:none;
    }

    div.stButton > button:hover {
        background-color:#ff4b4b;
        color:white;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>

    /* Prediction Result Card */
    .result-card {
        background: linear-gradient(135deg,#16a085,#27ae60);
        padding:30px;
        border-radius:14px;
        text-align:center;
        font-size:30px;
        font-weight:bold;
        color:white;
        margin-top:15px;
        margin-bottom:15px;
        box-shadow:0px 6px 18px rgba(0,0,0,0.25);
    }

    /* Result Title */
    .result-title{
        font-size:25px;
        font-weight:800;
        opacity:0.9;
    }

    </style>
    """, unsafe_allow_html=True)

    # ================= HEADER =================
    # ================= PROFESSIONAL HEADER =================

    st.markdown("""
        <style>

        /* Title Style */
        .header-title{
            font-size:28px;
            font-weight:700;
            color:#00C8FF;
            margin-bottom:5px;
        }

        /* Subtitle */
        .header-sub{
            font-size:16px;
            color:#bbbbbb;
        }

        /* Image Styling */
        .header-img img{
            border-radius:12px;
        }

        </style>
        """, unsafe_allow_html=True)

    col_img, col_title = st.columns([1.2,2])

    # IMAGE (always visible)
    with col_img:
        st.image("helathcare.png", use_container_width=True)

    # TITLE
    with col_title:
        st.markdown('<div class="header-title">🏥 Medical Insurance Cost Prediction</div>', unsafe_allow_html=True)

        st.markdown(
        '<div class="header-sub">'
        'Predict healthcare insurance charges using demographic and lifestyle factors powered by Machine Learning.'
        '</div>',
        unsafe_allow_html=True
        )

    st.markdown("---")

    # ================= INPUTS =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Patient Metrics")
        age = st.slider("Age",18,100,30)
        bmi = st.slider("BMI",10.0,50.0,25.0)
        children = st.slider("Number of Children",0,5,0)

    with col2:
        st.subheader("🧾 Patient Details")
        sex = st.selectbox("Sex",["Female","Male"])
        smoker = st.selectbox("Smoker",["No","Yes"])
        region = st.selectbox(
            "Region",
            ["Northeast","Northwest","Southeast","Southwest"]
        )

    # ================= ENCODING =================
    sex_val = 1 if sex=="Male" else 0
    smoker_val = 1 if smoker=="Yes" else 0

    region_map={
        "Northeast":0,
        "Northwest":1,
        "Southeast":2,
        "Southwest":3
    }

    region_val = region_map[region]

    st.markdown("---")

    # ================= PREDICTION OUTPUT =================
    if st.button("💰 Predict Insurance Cost"):

        input_data = np.array(
            [[age,sex_val,bmi,children,smoker_val,region_val]]
        )

        prediction = model.predict(input_data)[0]

        # BIG NOTICEABLE RESULT CARD
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">Estimated Medical Insurance Cost</div>
            ₹ {prediction:,.2f}
        </div>
        """, unsafe_allow_html=True)

        # RESULT TABLE
        result_df = pd.DataFrame({
            "Age":[age],
            "Sex":[sex],
            "BMI":[bmi],
            "Children":[children],
            "Smoker":[smoker],
            "Region":[region],
            "Predicted Medical Cost (₹)":[f"{prediction:,.2f}"]
        })

        st.markdown("### 📋 Prediction Summary")
        st.table(result_df)

    st.markdown("---")
    st.info("Prediction is based on Historical data.")