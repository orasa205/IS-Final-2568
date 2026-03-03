import streamlit as st

st.set_page_config(page_title="NN Model Explanation", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main { background: linear-gradient(to bottom, #f8f9fa, #e9ecef); }
    h1 { color: #2c3e50; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    h2 { color: #34495e; border-bottom: 2px solid #9b59b6; padding-bottom: 10px; }
    h3 { color: #8e44ad; }
    .theory-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .formula-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #9b59b6;
        margin: 10px 0;
    }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50; border: none; border-radius: 10px; padding: 10px 25px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Neural Network - Model Explanation")
st.markdown("---")

st.markdown("""
<div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
            padding: 20px; border-radius: 15px; color: #2c3e50; margin-bottom: 20px;">
    <h2 style="color: #2c3e50; border: none;">🎯 Learning Objectives</h2>
    <p style="font-size: 16px;">Understand neural network concepts and how they classify iris flower species</p>
</div>
""", unsafe_allow_html=True)

with st.expander("📋 1. Dataset: iris_clean.csv", expanded=True):
    st.header("1. Dataset: iris_clean.csv")
    st.write("The Iris dataset contains 150 samples of iris flowers with 4 features measuring sepal and petal dimensions.")
    
    st.subheader("Features:")
    st.write("- Sepal.Length, Sepal.Width, Petal.Length, Petal.Width")
    
    st.subheader("Target:")
    st.write("- Species: Setosa, Versicolor, Virginica")

with st.expander("🔧 2. Data Preprocessing", expanded=True):
    st.header("2. Data Preprocessing")
    
    st.subheader("2.1 Label Encoding")
    st.write("Setosa → 0, Versicolor → 1, Virginica → 2")
    
    st.subheader("2.2 One-Hot Encoding")
    st.write("Converts labels to: Setosa=[1,0,0], Versicolor=[0,1,0], Virginica=[0,0,1]")
    
    st.subheader("2.3 Feature Scaling")
    st.write("StandardScaler: $z = (x - μ) / σ$")
    
    st.subheader("2.4 Train/Test Split")
    st.write("Training: 80% | Test: 20% | Random State: 42")

with st.expander("🧮 3. Neural Network Theory", expanded=True):
    st.header("3. Neural Network Theory")
    
    st.subheader("3.1 Artificial Neurons")
    st.write("Basic unit that receives inputs, processes them with weights and bias, and produces output through activation.")
    
    st.subheader("3.2 Activation Functions")
    st.markdown("""
    <div class="formula-box">
    <b>ReLU:</b> f(x) = max(0, x) <br>
    <b>Softmax:</b> f(x_i) = e^x_i / Σe^x_j
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("3.3 Forward Propagation")
    st.write("Process of passing input through layers: a^(l) = f(W^(l) a^(l-1) + b^(l))")
    
    st.subheader("3.4 Loss Function")
    st.write("Categorical Crossentropy: L = -Σ y_c log(ŷ_c)")
    
    st.subheader("3.5 Backpropagation")
    st.write("Algorithm to update weights using gradient descent and chain rule.")

with st.expander("🏗️ 4. Model Architecture Design", expanded=True):
    st.header("4. Model Architecture Design")
    
    st.subheader("4.1 Input Layer")
    st.write("4 neurons (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width)")
    
    st.subheader("4.2 Hidden Layers")
    st.write("- Layer 1: 64 neurons, ReLU, Dropout 0.2")
    st.write("- Layer 2: 32 neurons, ReLU, Dropout 0.2")
    
    st.subheader("4.3 Output Layer")
    st.write("3 neurons (one per class), Softmax activation")
    
    st.subheader("4.4 Optimizer: Adam")
    st.write("Learning rate: 0.001, combines momentum and adaptive learning rates")

with st.expander("📊 5. Model Evaluation", expanded=True):
    st.header("5. Model Evaluation")
    
    st.subheader("5.1 Accuracy")
    st.write("Proportion of correct predictions: Accuracy = (TP+TN)/(TP+TN+FP+FN)")
    
    st.subheader("5.2 Confusion Matrix")
    st.write("Matrix showing true vs predicted labels with Precision, Recall, F1-Score")

with st.expander("📚 6. Dataset Source Reference", expanded=True):
    st.header("6. Dataset Source Reference")
    st.markdown("""
    **Dataset:** Iris Species  
    **Source:** Kaggle  
    **URL:** https://www.kaggle.com/datasets
    
    > Fisher, R.A. (1988). *Iris* [Data set]. Originally published in Fisher, R.A. (1936) 
    > "The use of multiple measurements in taxonomic problems", Annals of Eugenics.
    """)

st.markdown("---")
st.info("🚀 Navigate to **Neural Network - Model Testing** to train and test the neural network!")
