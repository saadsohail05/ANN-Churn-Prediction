import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load the model and preprocessing objects
def load_model():
    model = tf.keras.models.load_model("model.h5")
    labelencoder = pickle.load(open('labelencoder.pkl', 'rb'))
    onehotencoder = pickle.load(open('onehotencoder.pkl', 'rb'))
    sc = pickle.load(open('scaler.pkl', 'rb'))
    return model, labelencoder, onehotencoder, sc

model, labelencoder, onehotencoder, sc = load_model()

# Set page config
st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict the likelihood of a customer churning based on their information.")

# Input form
st.subheader("Customer Information")
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (years)", 0, 10, 3)

with col2:
    balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
    num_products = st.slider("Number of Products", 1, 4, 2)
    has_credit_card = st.checkbox("Has Credit Card")
    is_active_member = st.checkbox("Is Active Member")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Prediction function
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])
    input_df["Gender"] = labelencoder.transform(input_df["Gender"])
    
    geo_encoded = onehotencoder.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))
    
    input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
    input_scaled = sc.transform(input_df)
    
    prediction_prob = model.predict(input_scaled)[0][0]
    return prediction_prob

# Make prediction
if st.button("Predict Churn"):
    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary
    }
    
    churn_probability = predict_churn(input_data)
    
    # Display result
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_probability,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability"},
            gauge = {
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 0.75], 'color': "yellow"},
                    {'range': [0.75, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if churn_probability > 0.5:
            st.error(f"⚠️ The customer is likely to churn (Probability: {churn_probability:.2f})")
            st.markdown("### Recommendations:")
            st.markdown("1. Offer personalized retention incentives")
            st.markdown("2. Conduct a customer satisfaction survey")
            st.markdown("3. Provide additional value-added services")
        else:
            st.success(f"✅ The customer is not likely to churn (Probability: {churn_probability:.2f})")
            st.markdown("### Recommendations:")
            st.markdown("1. Continue providing excellent service")
            st.markdown("2. Offer loyalty rewards to maintain satisfaction")
            st.markdown("3. Regularly check-in for feedback and suggestions")

# Feature importance (placeholder - you would need to calculate this separately)
st.subheader("Feature Importance")
feature_importance = {
    'CreditScore': 0.15,
    'Age': 0.2,
    'Tenure': 0.1,
    'Balance': 0.18,
    'NumOfProducts': 0.12,
    'HasCrCard': 0.05,
    'IsActiveMember': 0.15,
    'EstimatedSalary': 0.05
}

fig = go.Figure(data=[go.Bar(
    x=list(feature_importance.keys()),
    y=list(feature_importance.values()),
    marker_color='lightblue'
)])
fig.update_layout(
    title="Feature Importance in Churn Prediction",
    xaxis_title="Features",
    yaxis_title="Importance",
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# Add some context about the model
st.subheader("About the Model")
st.markdown("""
This churn prediction model uses an Artificial Neural Network (ANN) trained on historical customer data. 
It takes into account various factors such as credit score, age, balance, and product usage to estimate 
the likelihood of a customer leaving the bank. The model's performance metrics and limitations should be 
considered when interpreting results.
""")

# Disclaimer
st.caption("Disclaimer: This is a predictive model and should be used as a tool to support decision-making, not as a sole determinant.")