import streamlit as st
import pandas as pd
import cloudpickle
import os

# Load dataset to infer feature columns
DATA_PATH = "daily_price.csv"
MODEL_PATH = "commodities_price.pkl"

def load_data():
    return pd.read_csv(DATA_PATH)

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return cloudpickle.load(f)

# Main Streamlit App

st.title("🌾 Commodity Price Predictor")

st.markdown("This app predicts the **price of a commodity** based on input features using a pre-trained model.")

# Load data and model
try:
    data = load_data()
    model = load_model()
except FileNotFoundError:
    st.error("Required files not found. Please ensure 'daily_price.csv' and 'model.pkl' exist in the same folder.")

# Feature selection (excluding target column)
target_col = 'price' if 'price' in data.columns else data.columns[-1]  # guessing target column
feature_cols = [col for col in data.columns if col != target_col]

st.header("📥 Input Commodity Data")

# Create input fields dynamically
input_data = {}
for col in feature_cols:
    dtype = data[col].dtype

    if pd.api.types.is_numeric_dtype(dtype):
        value = st.number_input(f"Enter {col}", value=float(data[col].mean()))
    else:
        value = st.selectbox(f"Select {col}", options=sorted(data[col].dropna().unique()))
    input_data[col] = value

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("🚀 Predict Price"):
    try:
        prediction = model.predict(input_df)
        st.success(f"📈 Predicted Commodity Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
