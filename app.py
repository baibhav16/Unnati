# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and encoder
model = joblib.load("model/app_id_classifier.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

# Get feature names used during training
try:
    trained_features = model.feature_names_in_
except AttributeError:
    trained_features = joblib.load("model/feature_names.pkl")

# Streamlit app
st.set_page_config(page_title="Traffic Classifier", layout="wide")
st.title("🔍 AI-Powered Network Traffic Classifier")

uploaded_file = st.file_uploader("📁 Upload a CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Drop empty columns and handle NaNs/Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, thresh=len(df) * 0.9, inplace=True)
        df.dropna(inplace=True)

        # Select only numeric columns
        X = df.select_dtypes(include=[np.number])

        # Align with training features
        missing_cols = set(trained_features) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[trained_features]  # Ensure column order

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        y_pred = model.predict(X_scaled)
        labels = label_encoder.inverse_transform(y_pred)

        df["Prediction"] = labels
        st.success("✅ Prediction completed!")
        st.dataframe(df)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="classified_traffic.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
