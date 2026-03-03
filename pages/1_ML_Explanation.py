import streamlit as st

st.set_page_config(page_title="ML Model Explanation", page_icon="📖", layout="wide")

st.markdown("""
<style>
    .main { background: linear-gradient(to bottom, #f8f9fa, #e9ecef); }
    h1 { color: #2c3e50; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h3 { color: #2980b9; }
    .theory-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .formula-box {
        background: #ecf0f1;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }
    .highlight { background: #fff3cd; padding: 2px 8px; border-radius: 4px; }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 10px; padding: 10px 25px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("📖 Machine Learning - Model Explanation")
st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; border-radius: 15px; color: white; margin-bottom: 20px;">
    <h2 style="color: white; border: none;">🎯 Learning Objectives</h2>
    <p style="font-size: 16px;">Understand the theory behind regression algorithms and how they predict student GPA</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📋 1. Data Preparation", expanded=True):
    st.header("1. Data Preparation")
    
    st.subheader("1.1 Dataset: std_clean.csv")
    st.write("The Student Performance Dataset contains information about students' academic performance and various factors that may influence their Grade Point Average (GPA).")
    
    st.subheader("1.2 Data Cleaning")
    st.write("""
    - **Removing duplicates**: Eliminate repeated records
    - **Handling inconsistencies**: Standardize data formats
    - **Outlier treatment**: Identify and handle extreme values
    """)
    
    st.subheader("1.3 Handling Missing Values")
    st.write("""
    - **Mean imputation**: For numerical features
    - **Mode imputation**: For categorical features  
    - **Row deletion**: For excessive missing data
    """)
    
    st.subheader("1.4 Encoding Categorical Variables")
    st.write("Using **Label Encoding**: Sex, Additional_Work, Sports_activity, etc. → 0/1")
    
    st.subheader("1.5 Feature Scaling")
    st.write("Using **StandardScaler**: $z = (x - μ) / σ$")
    
    st.subheader("1.6 Train/Test Split")
    st.write("Training: 80% | Test: 20% | Random State: 42")

with st.expander("📊 2. Algorithm Theory", expanded=True):
    st.header("2. Algorithm Theory")
    
    st.subheader("2.1 Linear Regression")
    st.markdown("""
    <div class="formula-box">
    <b>Formula:</b><br>
    $$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + ... + \\beta_n X_n + \\epsilon$$
    </div>
    """, unsafe_allow_html=True)
    st.write("Models the relationship between dependent and independent variables using a linear equation.")
    
    st.subheader("2.2 Random Forest Regressor")
    st.write("Ensemble method combining multiple decision trees with bagging and feature randomness.")
    
    st.subheader("2.3 K-Nearest Neighbors (KNN)")
    st.write("Instance-based learning using distance measures (Euclidean) to find K nearest neighbors.")
    
    st.subheader("2.4 Voting Regressor")
    st.write("Ensemble averaging of multiple regression models for more robust predictions.")

with st.expander("⚙️ 3. Model Development Process", expanded=True):
    st.header("3. Model Development Process")
    
    st.subheader("3.1 Feature Selection")
    st.write("Features: Student_Age, Sex, Additional_Work, Sports_activity, Transportation, Weekly_Study_Hours, Reading, Notes, Listening_in_Class, Project_work, Attendance Percentage")
    
    st.subheader("3.2 Model Training")
    st.write("Each model trained on 80% of data with default hyperparameters.")
    
    st.subheader("3.3 Model Evaluation (R² Score)")
    st.markdown("""
    <div class="formula-box">
    <b>R² Formula:</b><br>
    $$R^2 = 1 - \\frac{SS_{res}}{SS_{tot}} = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$
    </div>
    """, unsafe_allow_html=True)
    st.write("Measures how well the model explains variance in the target variable.")
    
    st.subheader("3.4 Performance Comparison")
    st.write("Compare models using R² Score, MAE, and RMSE.")

with st.expander("📚 4. Data Source Reference", expanded=True):
    st.header("4. Data Source Reference")
    st.markdown("""
    **Dataset:** Student Performance Factors  
    **Source:** Kaggle  
    **URL:** https://www.kaggle.com/datasets
    
    > Author. (Year). *Student Performance Factors Dataset*. Kaggle.
    """)

st.markdown("---")
st.info("🚀 Navigate to **Machine Learning - Model Testing** to train and test these models!")
