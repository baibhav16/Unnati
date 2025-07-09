import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model, scaler, and label encoder using pickle
with open("model/app_id_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get feature names from training model
expected_features = scaler.feature_names_in_

st.set_page_config(page_title="Network Traffic Classifier", layout="wide")
st.title("🚦 Network Traffic Classification & Anomaly Detection")

st.write("Upload a CSV file with extracted features to predict application ID and detect anomalies.")

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)

        # Clean columns
        df.columns = df.columns.str.strip()

        # Handle missing/extra features
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0

        df = df[expected_features]

        # Replace inf/-inf and drop rows with NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Scale
        X_scaled = scaler.transform(df)

        # Predict
        predictions = model.predict(X_scaled)
        prediction_labels = label_encoder.inverse_transform(predictions)

        # Show results
        df['Prediction'] = prediction_labels
        st.success("✅ Prediction complete!")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("Please upload a file to start prediction.")
