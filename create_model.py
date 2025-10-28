import pandas as pd
import numpy as np
import joblib  # <-- We use joblib to save the model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os


print("Loading all training data...")
if not os.path.exists("training_data.csv"):
    print("Error: 'training_data.csv' not found.")
    print("Please make sure 'training_data.csv' is in this directory.")
    exit()

train_df = pd.read_csv("training_data.csv")
print(f"Successfully loaded training_data.csv (Shape: {train_df.shape})")

target_column = 'prali_fire'

def preprocess_features(df):
    """Applies minimal feature engineering."""
    df = df.copy()
    df['acq_hour'] = df['acq_time'].astype(str).str.zfill(4).str[:2].astype(int)
    return df

print("Applying preprocessing...")
train_df_processed = preprocess_features(train_df)

new_numeric_features = [
    'brightness', 'scan', 'track', 'confidence', 'bright_t31', 
    'frp', 'type', 'acq_hour'
]

new_categorical_features = [
    'satellite', 'instrument', 'daynight'
]

all_features = new_numeric_features + new_categorical_features
print(f"Training model with {len(all_features)} features (No location/date).")

X_train = train_df_processed[all_features]
y_train = train_df_processed[target_column]

print("Preprocessing complete.")


numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, new_numeric_features),
        ('cat', categorical_transformer, new_categorical_features)
    ])

model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

print("\nTraining the final model on all training data...")
pipeline.fit(X_train, y_train)
print("Final model training complete.")

model_filename = 'prali_fire_model.pkl'
print(f"\nSaving the entire model pipeline to '{model_filename}'...")
joblib.dump(pipeline, model_filename)
print("Model saved successfully.")
print("You can now run 'streamlit run app.py'")