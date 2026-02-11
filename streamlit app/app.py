import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("insurance_model.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)
# Banner Image
st.image("helathcare.png")

# Custom CSS Styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg,#2E86C1,#48C9B0);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 32px;
    font-weight: bold;
}
.sub-header {
    text-align:center;
    font-size:18px;
    color:gray;
    margin-bottom:20px;
}
.result-box {
    background-color:#D4EFDF;
    padding:25px;
    border-radius:12px;
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:#1E8449;
}
.summary-card {
    background-color:#EBF5FB;
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🏥 Medical Insurance Cost Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict healthcare insurance charges using Machine Learning</div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
st.sidebar.header("🧾 Enter Patient Details")

age = st.sidebar.slider("Age", 18, 100, 25)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 0)

sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

# Encoding
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_mapping = {
    "Northeast": 0,
    "Northwest": 1,
    "Southeast": 2,
    "Southwest": 3
}
region_val = region_mapping[region]

# Layout columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Patient Summary")
    st.write(f"**Age:** {age}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Children:** {children}")
    st.write(f"**Sex:** {sex}")
    st.write(f"**Smoker:** {smoker}")
    st.write(f"**Region:** {region}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 🤖 Prediction")

    if st.button("Predict Insurance Cost 💰"):
        input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
        prediction = model.predict(input_data)

        st.markdown(
            f'<div class="result-box">Estimated Insurance Cost: ₹ {prediction[0]:,.2f}</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("⚠️ Prediction is based on a trained Machine Learning model and may vary from actual insurance charges.")
