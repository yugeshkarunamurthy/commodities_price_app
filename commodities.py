import streamlit as st
import pandas as pd
import joblib
import os
 
data = pd.read_csv("daily_price.csv")

model = joblib.load("commodities_price.pkl")

# Main Streamlit App

st.title("🌾 Commodity Price Predictor")

st.markdown("This app predicts the **price of a commodity** based on input features using a pre-trained model.")

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
