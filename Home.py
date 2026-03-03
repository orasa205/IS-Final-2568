import streamlit as st

st.set_page_config(
    page_title="Home - ML & NN Application",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background: linear-gradient(to right, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2, h3 {
        color: #34495e;
    }
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
    }
    .stMetric {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Machine Learning & Neural Network Application")
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    ## Welcome to the ML & Neural Network Application
    
    This application demonstrates both **Machine Learning** and **Deep Learning** techniques 
    with interactive model training and prediction capabilities.
    """)

with col2:
    st.markdown("### 📁 Navigation")
    st.info("Use the **sidebar** to navigate between pages!")

st.markdown("---")

st.markdown("""
## 📚 Application Overview

| Section | Pages | Description |
|---------|-------|-------------|
| **Machine Learning** | Model Explanation | Learn about regression algorithms |
| | Model Testing | Train & test regression models (GPA Prediction) |
| **Neural Network** | Model Explanation | Learn about deep learning concepts |
| | Model Testing | Train & test neural networks (Iris Classification) |
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 25px; border-radius: 15px; color: white;">
        <h3 style="color: white; margin-bottom: 15px;">📊 Machine Learning</h3>
        <p style="font-size: 16px;">Regression models to predict student GPA based on various factors:</p>
        <ul style="font-size: 14px;">
            <li>Linear Regression</li>
            <li>Random Forest Regressor</li>
            <li>K-Nearest Neighbors</li>
            <li>Voting Regressor</li>
        </ul>
        <p><strong>Dataset:</strong> Student Performance (std_clean.csv)</p>
        <p><strong>Target:</strong> GPA (Grade Point Average)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 25px; border-radius: 15px; color: white;">
        <h3 style="color: white; margin-bottom: 15px;">🧠 Neural Network</h3>
        <p style="font-size: 16px;">Deep learning model to classify iris flower species:</p>
        <ul style="font-size: 14px;">
            <li>Multi-layer Neural Network</li>
            <li>ReLU Activation</li>
            <li>Softmax Output</li>
            <li>Adam Optimizer</li>
        </ul>
        <p><strong>Dataset:</strong> Iris Flowers (iris_clean.csv)</p>
        <p><strong>Target:</strong> Species Classification</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("### 🚀 Quick Start")
st.info("""
1. **Start with Model Explanation pages** to learn about the algorithms
2. **Go to Model Testing pages** to train and test the models
3. **Make predictions** using the trained models
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 20px;">
    <p>Built with ❤️ using Streamlit, Scikit-learn, and TensorFlow</p>
    <p>📚 Educational Purpose Only</p>
</div>
""", unsafe_allow_html=True)
