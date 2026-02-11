import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("insurance_model.pkl", "rb"))

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Prediction")
st.write("Enter patient details to predict medical insurance charges.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)

# Dropdowns
sex = st.selectbox("Sex", ["Female", "Male"])
smoker = st.selectbox("Smoker", ["No", "Yes"])
region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# Encoding inputs
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0

region_mapping = {
    "Northeast": 0,
    "Northwest": 1,
    "Southeast": 2,
    "Southwest": 3
}
region_val = region_mapping[region]

# Prediction
if st.button("Predict Insurance Cost"):

    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
    prediction = model.predict(input_data)

    st.success(f"Estimated Medical Insurance Cost: ₹ {prediction[0]:,.2f}")
    
st.caption("Prediction is based on trained machine learning model and may vary from actual insurance charges.")
