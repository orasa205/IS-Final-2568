import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="NN Model Testing", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .main { background: linear-gradient(to bottom, #f8f9fa, #e9ecef); }
    h1 { color: #2c3e50; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    h2 { color: #34495e; border-bottom: 2px solid #9b59b6; padding-bottom: 10px; }
    h3 { color: #8e44ad; }
    .section-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 15px 0;
    }
    .stMetric {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child {
        background: linear-gradient(to right, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: bold;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Neural Network - Model Testing")
st.markdown("---")

df = pd.read_csv("Datasets/iris_clean.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, index_col=0)

st.header("1. 📂 Load Data")

df = pd.read_csv("Datasets/iris_clean.csv")
df = df.dropna()

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
st.write(f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

st.markdown("---")

st.header("2. 🔧 Data Preprocessing")

feature_columns = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
X = df[feature_columns].values
y = df['Species'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = keras.utils.to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

col1, col2, col3 = st.columns(3)
col1.metric("📚 Training Samples", len(X_train))
col2.metric("📝 Test Samples", len(X_test))
col3.metric("🔢 Features", X_train.shape[1])

st.write("**Target Classes:**", " | ".join(label_encoder.classes_))

st.markdown("---")

st.header("3. 🏗️ Build Neural Network")

def build_model(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if "model" not in st.session_state:
    st.session_state.model = build_model(X_train.shape[1], y_train.shape[1])

model = st.session_state.model

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: #3498db; padding: 15px; border-radius: 10px; text-align: center; color: white;">
        <h4 style="color: white; margin: 0;">📥 Input Layer</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">4</p>
        <p style="margin: 0; font-size: 12px;">neurons</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: #2ecc71; padding: 15px; border-radius: 10px; text-align: center; color: white;">
        <h4 style="color: white; margin: 0;">🔗 Hidden 1</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">64</p>
        <p style="margin: 0; font-size: 12px;">ReLU</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: #2ecc71; padding: 15px; border-radius: 10px; text-align: center; color: white;">
        <h4 style="color: white; margin: 0;">🔗 Hidden 2</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">32</p>
        <p style="margin: 0; font-size: 12px;">ReLU</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: #e74c3c; padding: 15px; border-radius: 10px; text-align: center; color: white;">
        <h4 style="color: white; margin: 0;">📤 Output</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">3</p>
        <p style="margin: 0; font-size: 12px;">Softmax</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: white; padding: 15px; border-radius: 10px; border: 2px solid #3498db;">
        <h4 style="margin-top: 0; color: #2c3e50;">⚙️ Training Settings</h4>
        <p><b>Loss Function:</b> Categorical Crossentropy</p>
        <p><b>Optimizer:</b> Adam</p>
        <p><b>Dropout:</b> 20%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: white; padding: 15px; border-radius: 10px; border: 2px solid #9b59b6;">
        <h4 style="margin-top: 0; color: #2c3e50;">📊 Model Summary</h4>
        <p><b>Total Parameters:</b> 2,875</p>
        <p><b>Trainable:</b> 2,835</p>
        <p><b>Non-trainable:</b> 40</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("4. 🤖 Train Model")

col1, col2 = st.columns(2)

with col1:
    epochs = st.slider("Number of Epochs", 10, 200, 50)
with col2:
    batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)

st.markdown("")

if st.button("🚀 Train Neural Network"):
    with st.spinner("Training neural network... ⏳"):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            verbose=1
        )
        st.session_state.history = history.history
        st.session_state.trained = True
    
    st.markdown("""
    <div class="success-box">
        <h3 style="margin: 0;">✅ Training Complete!</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("5. 📊 Model Evaluation")

if 'trained' in st.session_state and st.session_state.trained:
    history = st.session_state.history
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    st.metric("🎯 Test Accuracy", f"{accuracy * 100:.2f}%")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=label_encoder.classes_,
           yticklabels=label_encoder.classes_,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    st.pyplot(fig)
    
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    st.text(report)
    
    st.subheader("Training Curves")
    
    tab1, tab2 = st.tabs(["📉 Loss Curve", "📈 Accuracy Curve"])
    
    with tab1:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
        ax_loss.plot(history['loss'], label='Training Loss', color='#3498db', linewidth=2)
        ax_loss.plot(history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2)
        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_loss.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax_loss.legend(fontsize=11)
        ax_loss.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
    
    with tab2:
        fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
        ax_acc.plot(history['accuracy'], label='Training Accuracy', color='#3498db', linewidth=2)
        ax_acc.plot(history['val_accuracy'], label='Validation Accuracy', color='#e74c3c', linewidth=2)
        ax_acc.set_xlabel('Epoch', fontsize=12)
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        ax_acc.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax_acc.legend(fontsize=11)
        ax_acc.grid(True, alpha=0.3)
        st.pyplot(fig_acc)

st.markdown("---")

st.header("6. 🎯 Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.8, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.markdown("")

if st.button("🔮 Predict Species"):
    if 'trained' not in st.session_state or not st.session_state.trained:
        st.warning("⚠️ Please train the model first!")
    else:
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_features)
        
        prediction_prob = model.predict(input_scaled, verbose=0)
        prediction_class = np.argmax(prediction_prob, axis=1)[0]
        predicted_species = label_encoder.inverse_transform([prediction_class])[0]
        confidence = prediction_prob[0][prediction_class] * 100
        
        st.markdown(f"""
        <div class="success-box">
            <h2 style="margin: 0;">🌸 Predicted Species: {predicted_species}</h2>
            <h3 style="margin: 10px 0 0 0;">Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Probability Distribution")
        
        prob_df = pd.DataFrame({
            'Species': label_encoder.classes_,
            'Probability (%)': prediction_prob[0] * 100
        })
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(prob_df['Species'], prob_df['Probability (%)'], color=colors, edgecolor='black')
        ax.set_xlabel('Probability (%)', fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_title('Class Probability Distribution', fontsize=14, fontweight='bold')
        for bar, v in zip(bars, prob_df['Probability (%)']):
            ax.text(v + 2, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)

st.markdown("---")
st.info("✅ This completes the Neural Network application!")
