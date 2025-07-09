import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Step 1: Load dataset ===
df = pd.read_csv("train1.csv")  # Update this with your dataset path

# Clean column names
df.columns = df.columns.str.strip()

# Check for 'Label'
if 'Label' not in df.columns:
    raise Exception("❌ 'Label' column is required in the dataset.")

# === Step 2: Preprocess ===
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, thresh=len(df) * 0.9, inplace=True)  # Drop mostly empty columns
df.dropna(inplace=True)  # Drop remaining rows with NaN

# === Step 3: Split features and label ===
y = df['Label']
X = df.drop(columns=['Label'])

# Keep only numeric features
X = X.select_dtypes(include=[np.number])
y = y[X.index]  # Align

# === Step 4: Encode target labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === Step 5: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Step 6: Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 7: Hyperparameter tuning ===
param_grid = {
    "n_estimators": [100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# === Step 8: Evaluation ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Step 9: Save model, scaler, and encoder using pickle ===
os.makedirs("model", exist_ok=True)

with open("model/app_id_classifier.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("📦 Model, scaler, and label encoder saved to /model/")
