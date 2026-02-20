# 🏥 Medical Insurance Cost Prediction & Analysis using Machine Learning

---

## 🔍 Project Overview
Medical insurance cost estimation is an important problem in the healthcare and insurance industries.  
Insurance charges depend on several demographic and health-related factors such as **age, BMI, smoking status, number of dependents, gender, and region**.

This project focuses on **predicting medical insurance charges using machine learning** and providing **data-driven insights through interactive dashboards and a live prediction application**.

The project includes:
- End-to-end **Machine Learning pipeline**
- **Exploratory Data Analysis (EDA)**
- Multiple ML model training and comparison
- **Live insurance cost prediction app**
- **Interactive EDA dashboard**

---

## 🚀 Live Deployment
The project is deployed using **Streamlit Cloud**, providing both prediction and analytics capabilities.
### 🔗 Live Links
- **Medical Insurance DASHBOARD and PREDICTION app**  
   👉 https://medical-insurance-dashboard-and-prediction.streamlit.app/

- **Medical Insurance Prediction App**  
  👉 https://medical-insurance-cost-prediction-ml.streamlit.app/

- **Medical Insurance EDA Dashboard**  
  👉 https://medical-insurance-dashboard.streamlit.app/

- **GitHub Repository**  
  👉 https://github.com/pavan-ahire/Medical-Insurance-Cost-Prediction-ML

---

## 🎯 Objectives
- Analyze factors influencing medical insurance costs
- Perform in-depth exploratory data analysis
- Build and compare multiple ML models
- Predict insurance charges for new customers
- Provide insights through visual dashboards

---

## 💼 Business Problem & Impact
Insurance companies need accurate and data-driven methods to estimate customer premiums.

This project helps businesses to:
- Identify **key factors affecting insurance charges**
- Improve premium pricing strategies
- Reduce manual estimation errors
- Support data-driven decision making

This solution supports **insurance analysts, healthcare analysts, and business teams**.

---

## 🔄 End-to-End ML Pipeline
The project follows a **production-oriented ML workflow**:

1. Data collection & understanding  
2. Data cleaning & preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Feature engineering  
5. Model training  
6. Model comparison & evaluation  
7. Best model selection  
8. Model persistence (`.pkl`)  
9. Deployment using Streamlit  
10. Dashboard development for analysis  

---

## 🧠 Machine Learning Models Used
The following algorithms were implemented and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

📌 **Random Forest Regressor** was selected for deployment due to:
- Better performance compared to other models
- Ability to handle non-linear relationships
- Robustness and stability

---

## 📊 Exploratory Data Analysis (EDA)
EDA was performed to:
- Understand distribution of insurance charges
- Analyze relationships between features
- Identify correlations between variables
- Detect outliers and skewness
- Understand feature importance

Visualizations include:
- Distribution plots
- Correlation heatmaps
- Boxplots for outlier detection
- Feature vs charges analysis
- Histogram and density plots

---

## 🧩 Feature Engineering & Preprocessing
Key preprocessing steps:
- Encoding categorical variables
- Handling missing values (if any)
- Feature scaling (where required)
- Ensuring feature consistency during inference
- Saving preprocessing objects for deployment

This ensures **training and prediction pipelines remain identical**.

---

## 🧪 Model Evaluation Metrics
Models were evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation

These metrics help evaluate prediction accuracy and model reliability.

---

## 🖥️ Streamlit Prediction App Features
- Simple and clean UI
- Accepts customer details as input
- Predicts insurance charges instantly
- Real-time ML inference
- User-friendly interface

---

## 📈 Streamlit EDA Dashboard Features
- Interactive charts and visualizations
- Insurance charges distribution insights
- Feature correlation analysis
- Outlier visualization
- Responsive and interactive layout

---

## 🛠️ Technologies Used
- **Language**: Python  
- **Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit
  - joblib
- **Deployment**: Streamlit Cloud  
- **Version Control**: Git & GitHub  

---

## 📂 Project Folder Structure

```text
Medical-Insurance-Cost-Prediction-ML/
│
├── app/
│   └── Streamlit app components and UI logic
│
├── dashboard/
│   └── EDA dashboard scripts and visualizations
│
├── data/
│   └── Raw and cleaned datasets
│
├── notebooks/
│   └── EDA and model training notebooks
│
├── model/
│   └── Saved ML model artifacts
│
├── README.md
├── app.py
├── dashboard.py
├── insurance_model.pkl
├── requirements.txt
└── dataset.csv
```
---
## How to Run the Project Locally

Follow the steps below to run the project on your local machine:

### 1️⃣ Clone the Repository
```
git clone https://github.com/pavan-ahire/Medical-Insurance-Cost-Prediction-ML.git
cd Medical-Insurance-Cost-Prediction-ML

```

### Install Required Dependencies
- pip install -r requirements.txt
  
### Run streamlit prediction app
- streamlit run app.py
  
### Run Streamlit Dashboard
-streamlit run dashboard.py

---
## 🧠 Key Skills Demonstrated

- Machine Learning model development and evaluation
- Exploratory Data Analysis (EDA)
- Feature engineering and data preprocessing
- Model serialization and reuse (`.pkl` files)
- Deployment of ML models using Streamlit
- Dashboard creation for business insights
- End-to-end project implementation
- Version control using Git & GitHub
---
## 👨‍💻 Author

**Pavan Ahire**


 Aspiring Data Scientist | Machine Learning & Analytics Enthusiast
- [🔗 GitHub](https://github.com/pavan-ahire)
- [🔗 LinkedIn](https://www.linkedin.com/in/pavan-ahire-260940364/)
