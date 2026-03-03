import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Model Testing", page_icon="🧪", layout="wide")

st.markdown("""
<style>
    .main { background: linear-gradient(to bottom, #f8f9fa, #e9ecef); }
    h1 { color: #2c3e50; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h3 { color: #2980b9; }
    .section-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
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
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 10px; padding: 12px 30px; font-weight: bold; font-size: 16px;
    }
    div.stButton > button:hover {
        background: linear-gradient(to right, #43e97b 0%, #38f9d7 100%);
    }
    .dataframe { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Machine Learning - Model Testing")
st.markdown("---")

df = pd.read_csv("../Datasets/std_clean.csv")
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_PATH)

st.header("1. 📂 Data Loading & Filtering")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="section-box">
        <h3 style="margin-top: 0;">📤 Upload Dataset</h3>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

with col2:
    st.markdown("""
    <div class="section-box">
        <h3 style="margin-top: 0;">🎚️ Age Filter</h3>
    </div>
    """, unsafe_allow_html=True)
    min_age, max_age = int(df['Student_Age'].min()), int(df['Student_Age'].max())
    age_range = st.slider("Select Student Age Range", min_age, max_age, (min_age, max_age))

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom dataset loaded!")

filtered_df = df[(df['Student_Age'] >= age_range[0]) & (df['Student_Age'] <= age_range[1])]

st.subheader(f"📊 Filtered Dataset (Age {age_range[0]}-{age_range[1]})")
st.write(f"**Total Records:** {len(filtered_df)}")
st.dataframe(filtered_df.head(10), use_container_width=True)

st.markdown("---")

st.header("2. 📈 Data Visualization")

viz_type = st.selectbox("Select Visualization Type", ["Histogram", "Bar Chart"])

if viz_type == "Histogram":
    col1, col2 = st.columns(2)
    with col1:
        column = st.selectbox("Select Column", filtered_df.select_dtypes(include=[np.number]).columns)
    with col2:
        bins = st.slider("Number of Bins", 5, 50, 20)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(filtered_df[column].dropna(), bins=bins, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{column} Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

elif viz_type == "Bar Chart":
    column = st.selectbox("Select Column", ['Sex', 'Additional_Work', 'Sports_activity', 
                                             'Transportation', 'Reading', 'Notes', 
                                             'Listening_in_Class', 'Project_work'])
    fig, ax = plt.subplots(figsize=(10, 5))
    filtered_df[column].value_counts().plot(kind='bar', color=['#3498db', '#e74c3c'], ax=ax, edgecolor='white')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{column} Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

st.markdown("---")

st.header("3. 🤖 Model Training")

filtered_df = filtered_df.dropna()

categorical_cols = ['Sex', 'Additional_Work', 'Sports_activity', 'Transportation', 
                    'Reading', 'Notes', 'Listening_in_Class', 'Project_work']

df_encoded = filtered_df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

feature_cols = ['Student_Age', 'Sex', 'Additional_Work', 'Sports_activity', 
                'Transportation', 'Weekly_Study_Hours', 'Reading', 'Notes', 
                'Listening_in_Class', 'Project_work', 'Attendance Percentage']

X = df_encoded[feature_cols].values
y = df_encoded['GPA'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

col1, col2 = st.columns(2)
col1.metric("📚 Training Samples", len(X_train))
col2.metric("📝 Test Samples", len(X_test))

st.markdown("")

if st.button("🚀 Train All Models"):
    with st.spinner("Training models... ⏳"):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_r2 = r2_score(y_test, knn_pred)
        
        voting = VotingRegressor([
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('knn', KNeighborsRegressor(n_neighbors=5))
        ])
        voting.fit(X_train, y_train)
        voting_pred = voting.predict(X_test)
        voting_r2 = r2_score(y_test, voting_pred)
        
        st.session_state.models_trained = True
        st.session_state.results = {
            'Linear Regression': {'r2': lr_r2, 'pred': lr_pred, 'model': lr},
            'Random Forest': {'r2': rf_r2, 'pred': rf_pred, 'model': rf},
            'KNN': {'r2': knn_r2, 'pred': knn_pred, 'model': knn},
            'Voting Regressor': {'r2': voting_r2, 'pred': voting_pred, 'model': voting}
        }
    
    st.markdown("""
    <div class="success-box">
        <h3 style="color: white; margin: 0;">✅ Training Complete!</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.header("4. 📊 Model Performance")

if 'models_trained' in st.session_state and st.session_state.models_trained:
    results = st.session_state.results
    
    st.subheader("R² Score Comparison")
    
    r2_scores = {name: data['r2'] for name, data in results.items()}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax.bar(list(r2_scores.keys()), list(r2_scores.values()), color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_title('R² Score Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(r2_scores.values()) * 1.3)
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Good Fit (0.5)')
    ax.legend()
    for bar, score in zip(bars, r2_scores.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("📊 Detailed Results - All Metrics")
    
    model_names = list(results.keys())
    r2_values = [results[name]['r2'] for name in model_names]
    mae_values = [mean_absolute_error(y_test, results[name]['pred']) for name in model_names]
    rmse_values = [np.sqrt(mean_squared_error(y_test, results[name]['pred'])) for name in model_names]
    
    tab1, tab2, tab3 = st.tabs(["📈 R² Score", "📉 MAE", "📉 RMSE"])
    
    with tab1:
        fig_r2, ax_r2 = plt.subplots(figsize=(10, 5))
        colors_r2 = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars_r2 = ax_r2.bar(model_names, r2_values, color=colors_r2, edgecolor='black', linewidth=1.5)
        ax_r2.set_ylabel('R² Score', fontsize=12)
        ax_r2.set_title('R² Score by Model', fontsize=14, fontweight='bold', pad=20)
        ax_r2.set_ylim(0, 1.15)
        for bar, val in zip(bars_r2, r2_values):
            ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05, 
                    f'{val:.4f}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        plt.xticks(rotation=15, ha='right')
        ax_r2.grid(axis='y', alpha=0.3)
        st.pyplot(fig_r2)
    
    with tab2:
        fig_mae, ax_mae = plt.subplots(figsize=(10, 5))
        colors_mae = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
        bars_mae = ax_mae.bar(model_names, mae_values, color=colors_mae, edgecolor='black', linewidth=1.5)
        ax_mae.set_ylabel('MAE', fontsize=12)
        ax_mae.set_title('Mean Absolute Error by Model', fontsize=14, fontweight='bold', pad=20)
        ax_mae.set_ylim(0, max(mae_values) * 1.25)
        for bar, val in zip(bars_mae, mae_values):
            ax_mae.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, 
                    f'{val:.4f}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        plt.xticks(rotation=15, ha='right')
        ax_mae.grid(axis='y', alpha=0.3)
        st.pyplot(fig_mae)
    
    with tab3:
        fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
        colors_rmse = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
        bars_rmse = ax_rmse.bar(model_names, rmse_values, color=colors_rmse, edgecolor='black', linewidth=1.5)
        ax_rmse.set_ylabel('RMSE', fontsize=12)
        ax_rmse.set_title('Root Mean Squared Error by Model', fontsize=14, fontweight='bold', pad=20)
        ax_rmse.set_ylim(0, max(rmse_values) * 1.25)
        for bar, val in zip(bars_rmse, rmse_values):
            ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, 
                    f'{val:.4f}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')
        plt.xticks(rotation=15, ha='right')
        ax_rmse.grid(axis='y', alpha=0.3)
        st.pyplot(fig_rmse)
    
    st.markdown("---")
    
    st.subheader("📊 Combined Metrics Comparison (Line Chart)")
    
    fig_combined, ax_combined = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    
    line1, = ax_combined.plot(x, r2_values, marker='o', markersize=12, linewidth=3, label='R² Score', color='#3498db')
    line2, = ax_combined.plot(x, mae_values, marker='s', markersize=12, linewidth=3, label='MAE', color='#e74c3c')
    line3, = ax_combined.plot(x, rmse_values, marker='^', markersize=12, linewidth=3, label='RMSE', color='#2ecc71')
    
    for i, (r2, mae, rmse) in enumerate(zip(r2_values, mae_values, rmse_values)):
        ax_combined.annotate(f'{r2:.3f}', (i, r2), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#3498db')
        ax_combined.annotate(f'{mae:.3f}', (i, mae), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#e74c3c')
        ax_combined.annotate(f'{rmse:.3f}', (i, rmse), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#2ecc71')
    
    ax_combined.set_ylabel('Value', fontsize=12)
    ax_combined.set_xlabel('Model', fontsize=12)
    ax_combined.set_title('All Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    ax_combined.set_xticks(x)
    ax_combined.set_xticklabels(model_names, rotation=15, ha='right')
    ax_combined.legend(loc='upper right')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.set_ylim(0, max(max(r2_values), max(mae_values), max(rmse_values)) * 1.3)
    
    st.pyplot(fig_combined)
    
    st.markdown("---")
    
    st.subheader("📋 Summary Table")
    
    summary_data = pd.DataFrame({
        'Model': model_names,
        'R² Score': [f"{v:.4f}" for v in r2_values],
        'MAE': [f"{v:.4f}" for v in mae_values],
        'RMSE': [f"{v:.4f}" for v in rmse_values]
    })
    st.table(summary_data)

    st.subheader("Actual vs Predicted (Best Model)")
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_pred = results[best_model_name]['pred']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, best_pred, alpha=0.6, c='#3498db', s=80, edgecolors='white', linewidth=1)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual GPA', fontsize=12)
    ax.set_ylabel('Predicted GPA', fontsize=12)
    ax.set_title(f'Actual vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

st.markdown("---")

st.header("5. 🎯 Make a Prediction")

st.write("Enter feature values to predict GPA:")

col1, col2 = st.columns(2)

with col1:
    input_age = st.number_input("Student Age", 18, 30, 22)
    input_sex = st.selectbox("Sex", ["Male", "Female"])
    input_add_work = st.selectbox("Additional Work", ["Yes", "No"])
    input_sports = st.selectbox("Sports Activity", ["Yes", "No"])
    input_transport = st.selectbox("Transportation", ["Private", "Bus"])
    input_study_hours = st.number_input("Weekly Study Hours", 0, 25, 10)

with col2:
    input_reading = st.selectbox("Reading", ["Yes", "No"])
    input_notes = st.selectbox("Notes", ["Yes", "No"])
    input_listening = st.selectbox("Listening in Class", ["Yes", "No"])
    input_project = st.selectbox("Project Work", ["Yes", "No"])
    input_attendance = st.number_input("Attendance Percentage", 0, 100, 80)

def encode_input(value, column):
    return label_encoders[column].transform([value])[0]

if st.button("🎯 Predict GPA"):
    input_features = np.array([[
        input_age,
        encode_input(input_sex, 'Sex'),
        encode_input(input_add_work, 'Additional_Work'),
        encode_input(input_sports, 'Sports_activity'),
        encode_input(input_transport, 'Transportation'),
        input_study_hours,
        encode_input(input_reading, 'Reading'),
        encode_input(input_notes, 'Notes'),
        encode_input(input_listening, 'Listening_in_Class'),
        encode_input(input_project, 'Project_work'),
        input_attendance
    ]])
    
    input_scaled = scaler.transform(input_features)
    
    predictions = {}
    for name, data in results.items():
        pred = data['model'].predict(input_scaled)[0]
        predictions[name] = pred
    
    st.markdown("""
    <div class="success-box">
        <h3 style="color: white; margin: 0;">📊 GPA Predictions from All Models</h3>
    </div>
    """, unsafe_allow_html=True)
    
    model_names = list(predictions.keys())
    pred_values = list(predictions.values())
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        bars = ax.bar(model_names, pred_values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Predicted GPA', fontsize=14)
        ax.set_xlabel('Model', fontsize=14)
        ax.set_title('GPA Prediction Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(pred_values) * 1.3)
        ax.axhline(y=np.mean(pred_values), color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(pred_values):.2f}')
        ax.legend(fontsize=11)
        
        for bar, val in zip(bars, pred_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        plt.xticks(rotation=15, ha='right', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.write("### 📊 Values")
        for name, val in predictions.items():
            st.metric(name, f"{val:.2f}")
        
        st.markdown("---")
        st.write("**Average GPA:**")
        st.info(f"{np.mean(pred_values):.2f}")

st.markdown("---")
st.info("🚀 Navigate to **Neural Network - Model Explanation** to learn about neural networks!")
