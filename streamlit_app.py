"""
Telco Customer Churn Prediction - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    except FileNotFoundError:
        st.error("Data file not found. Please ensure WA_Fn-UseC_-Telco-Customer-Churn.csv is in the same directory.")
        return None
    
    # Data cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

@st.cache_data
def get_trained_model(df):
    """Train and return the model"""
    # Store customerID for later
    customer_ids = df['customerID'].copy()
    
    # Encode categorical columns
    le = LabelEncoder()
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'Churn', 'customerID']
    
    df_encoded = df.copy()
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Prepare features
    X = df_encoded.drop(['Churn', 'customerID'], axis=1)
    y = df_encoded['Churn']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, le, X.columns.tolist()

def main():
    st.title("📊 Telco Customer Churn Prediction")
    st.markdown("Predict which customers are likely to churn and take preventive measures!")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Data Overview", "EDA", "Prediction", "High Risk Customers"])
    
    if page == "Home":
        show_home(df)
    elif page == "Data Overview":
        show_data_overview(df)
    elif page == "EDA":
        show_eda(df)
    elif page == "Prediction":
        show_prediction(df)
    elif page == "High Risk Customers":
        show_high_risk(df)

def show_home(df):
    """Home page with summary"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        churn_count = df[df['Churn'] == 'Yes'].shape[0]
        st.metric("Churned Customers", churn_count)
    with col3:
        churn_rate = (churn_count / len(df)) * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("### Welcome to Customer Churn Predictor")
    st.markdown("""
    This app helps you:
    - 📈 Explore customer data and trends
    - 🔮 Predict churn probability for individual customers
    - ⚠️ Identify high-risk customers
    - 📊 Visualize key metrics
    """)
    
    st.markdown("### Quick Stats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Churn by Contract Type")
        contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        fig, ax = plt.subplots(figsize=(8, 5))
        contract_churn.plot(kind='bar', ax=ax, color=['#ff6b6b', '#ffa502', '#2ed573'])
        ax.set_ylabel('Churn Rate (%)')
        ax.set_xlabel('Contract Type')
        ax.set_title('Churn Rate by Contract Type')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("#### Churn by Internet Service")
        internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        fig, ax = plt.subplots(figsize=(8, 5))
        internet_churn.plot(kind='bar', ax=ax, color=['#ff6b6b', '#ffa502', '#2ed573'])
        ax.set_ylabel('Churn Rate (%)')
        ax.set_xlabel('Internet Service')
        ax.set_title('Churn Rate by Internet Service')
        plt.xticks(rotation=45)
        st.pyplot(fig)

def show_data_overview(df):
    """Show data overview"""
    st.header("📋 Data Overview")
    
    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))
    
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Unique Values': df.nunique().values,
        'Missing Values': df.isnull().sum().values
    })
    st.dataframe(col_info)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

def show_eda(df):
    """Show EDA visualizations"""
    st.header("📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Churn Distribution", "Demographics", "Services", "Billing"])
    
    with tab1:
        st.subheader("Churn Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            churn_counts = df['Churn'].value_counts()
            colors = ['#2ed573', '#ff6b6b']
            ax.pie(churn_counts, labels=['No Churn', 'Churn'], autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Churn Distribution')
            st.pyplot(fig)
        
        with col2:
            st.write(f"**Total Customers:** {len(df)}")
            st.write(f"**Churned:** {len(df[df['Churn'] == 'Yes'])}")
            st.write(f"**Retained:** {len(df[df['Churn'] == 'No'])}")
    
    with tab2:
        st.subheader("Demographics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            gender_churn = df.groupby(['gender', 'Churn']).size().unstack()
            gender_churn.plot(kind='bar', ax=ax, color=['#2ed573', '#ff6b6b'])
            ax.set_title('Churn by Gender')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Count')
            plt.xticks(rotation=0)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()
            senior_churn.plot(kind='bar', ax=ax, color=['#2ed573', '#ff6b6b'])
            ax.set_title('Churn by Senior Citizen Status')
            ax.set_xlabel('Senior Citizen')
            ax.set_ylabel('Count')
            plt.xticks(rotation=0)
            st.pyplot(fig)
    
    with tab3:
        st.subheader("Services Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()
            internet_churn.plot(kind='bar', ax=ax, color=['#2ed573', '#ff6b6b'])
            ax.set_title('Churn by Internet Service')
            ax.set_xlabel('Internet Service')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
            contract_churn.plot(kind='bar', ax=ax, color=['#2ed573', '#ff6b6b'])
            ax.set_title('Churn by Contract Type')
            ax.set_xlabel('Contract Type')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Billing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df[df['Churn'] == 'Yes']['MonthlyCharges'].hist(bins=30, ax=ax, color='#ff6b6b', alpha=0.7, label='Churned')
            df[df['Churn'] == 'No']['MonthlyCharges'].hist(bins=30, ax=ax, color='#2ed573', alpha=0.5, label='Retained')
            ax.set_title('Monthly Charges Distribution')
            ax.set_xlabel('Monthly Charges')
            ax.set_ylabel('Count')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            tenure_churn = df.groupby(pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '12-24', '24-48', '48-72']))['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            tenure_churn.plot(kind='bar', ax=ax, color='#ff6b6b')
            ax.set_title('Churn Rate by Tenure (months)')
            ax.set_xlabel('Tenure Range')
            ax.set_ylabel('Churn Rate (%)')
            plt.xticks(rotation=45)
            st.pyplot(fig)

def show_prediction(df):
    """Show prediction interface"""
    st.header("🔮 Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability")
    
    # Train model
    model, scaler, le, feature_names = get_trained_model(df)
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    
    # Additional services
    st.subheader("Additional Services")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    
    with col2:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    with col3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    # Billing
    st.subheader("Billing Information")
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    
    with col2:
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)
    
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Create input dataframe
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    if st.button("Predict Churn", type="primary"):
        # Create dataframe from input
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical columns
        categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod']
        
        for col in categorical_columns:
            if col in input_df.columns:
                le = LabelEncoder()
                # Fit on original data to get consistent encoding
                le.fit(df[col].astype(str))
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # Ensure columns match training data
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        
        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        # Display results
        st.markdown("### Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ Customer is likely to CHURN")
            else:
                st.success("✅ Customer is likely to STAY")
        
        with col2:
            st.metric("Churn Probability", f"{probability * 100:.1f}%")
        
        with col3:
            if probability >= 0.7:
                risk = "🔴 High Risk"
            elif probability >= 0.4:
                risk = "🟡 Medium Risk"
            else:
                risk = "🟢 Low Risk"
            st.markdown(f"**Risk Level:** {risk}")
        
        # Visual probability gauge
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(['Probability'], [probability], color='#ff6b6b' if probability > 0.5 else '#2ed573', height=0.3)
        ax.barh(['Probability'], [1 - probability], left=[probability], color='#2ed573' if probability > 0.5 else '#ff6b6b', height=0.3)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Churn Probability Gauge')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
        st.pyplot(fig)

def show_high_risk(df):
    """Show high risk customers"""
    st.header("⚠️ High Risk Customers")
    
    # Train model and get predictions
    model, scaler, le, feature_names = get_trained_model(df)
    
    # Prepare data for prediction
    df_encoded = df.copy()
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'Churn', 'customerID']
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            le.fit(df_encoded[col].astype(str))
            df_encoded[col] = le.transform(df_encoded[col].astype(str))
    
    X = df_encoded.drop(['Churn', 'customerID'], axis=1)
    X = X[feature_names]
    X_scaled = scaler.transform(X)
    
    # Get probabilities
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Add risk levels
    df_result = df.copy()
    df_result['Churn_Probability'] = probabilities
    
    def risk_category(prob):
        if prob >= 0.7:
            return "High Risk"
        elif prob >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"
    
    df_result['Risk_Level'] = df_result['Churn_Probability'].apply(risk_category)
    
    # Show risk distribution
    st.subheader("Risk Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = df_result['Risk_Level'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {'High Risk': '#ff6b6b', 'Medium Risk': '#ffa502', 'Low Risk': '#2ed573'}
        risk_counts.plot(kind='bar', ax=ax, color=[colors.get(x, '#999') for x in risk_counts.index])
        ax.set_title('Customer Risk Levels')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.write("### Risk Summary")
        high_risk = len(df_result[df_result['Risk_Level'] == 'High Risk'])
        medium_risk = len(df_result[df_result['Risk_Level'] == 'Medium Risk'])
        low_risk = len(df_result[df_result['Risk_Level'] == 'Low Risk'])
        
        st.metric("High Risk Customers", high_risk)
        st.metric("Medium Risk Customers", medium_risk)
        st.metric("Low Risk Customers", low_risk)
    
    # Show high risk customers
    st.subheader("High Risk Customers List")
    high_risk_df = df_result[df_result['Risk_Level'] == 'High Risk'].sort_values('Churn_Probability', ascending=False)
    
    # Display with probability
    display_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn_Probability', 'Risk_Level']
    st.dataframe(high_risk_df[display_cols].head(50))
    
    # Download option
    csv = high_risk_df.to_csv(index=False)
    st.download_button(
        label="Download High Risk Customers CSV",
        data=csv,
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()