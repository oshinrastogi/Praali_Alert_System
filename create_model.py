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

# --- ADD THIS CODE AFTER YOUR PREPROCESSING SECTION ---

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

print("\n--- Step 1: Correlation Analysis ---")
# Compute and plot correlation matrix (only for numeric features)
corr_matrix = train_df_processed[new_numeric_features + [target_column]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png")
print("Saved 'feature_correlation_heatmap.png'")

# --- Step 2: Split into Train / Validation / Test Sets ---
print("\n--- Step 2: Creating Train, Validation, and Test Sets ---")
X = train_df_processed[all_features]
y = train_df_processed[target_column]

# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Training Set: {X_train.shape}, Validation Set: {X_val.shape}, Test Set: {X_test.shape}")

# --- Step 3: Define Preprocessor (same as before) ---
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

# --- Step 4: Try Multiple Models ---
print("\n--- Step 3: Training Multiple Models ---")
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, clf in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# --- Step 5: Choose Best Model and Apply Hyperparameter Tuning ---
best_model_name = max(results, key=results.get)
print(f"\nBest base model: {best_model_name}")

if best_model_name == "Random Forest":
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
    base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
elif best_model_name == "Gradient Boosting":
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5]
    }
    base_model = GradientBoostingClassifier(random_state=42)
else:
    base_model = models[best_model_name]
    param_grid = {}

pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', base_model)
])

if param_grid:
    print("\n--- Step 4: Performing Hyperparameter Tuning ---")
    grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
else:
    best_pipeline = pipe.fit(X_train, y_train)

# --- Step 6: Final Evaluation on Test Set ---
print("\n--- Step 5: Final Model Evaluation on Test Data ---")
y_pred_test = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# --- Step 7: Save Final Tuned Model ---
joblib.dump(best_pipeline, 'prali_fire_model.pkl')
print("✅ Saved best tuned model as 'prali_fire_model.pkl'")

print("\nAll steps complete! Correlation, Model Comparison, and Tuning done successfully.")


# numeric_transformer = Pipeline(steps=[
#     ('scaler', StandardScaler())
# ])
# categorical_transformer = Pipeline(steps=[
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, new_numeric_features),
#         ('cat', categorical_transformer, new_categorical_features)
#     ])

# model = RandomForestClassifier(
#     n_estimators=100, 
#     random_state=42, 
#     n_jobs=-1,
#     class_weight='balanced'
# )

# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', model)
# ])

# print("\nTraining the final model on all training data...")
# pipeline.fit(X_train, y_train)
# print("Final model training complete.")

# model_filename = 'prali_fire_model.pkl'
# print(f"\nSaving the entire model pipeline to '{model_filename}'...")
# joblib.dump(pipeline, model_filename)
# print("Model saved successfully.")
# print("You can now run 'streamlit run app.py'")