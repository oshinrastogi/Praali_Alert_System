import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

#Load Data
print("Loading all training data...")
if not os.path.exists("training_data.csv"):
    print("Error: 'training_data.csv' not found.")
    exit()

train_df = pd.read_csv("training_data.csv")
print(f"Successfully loaded training_data.csv (Shape: {train_df.shape})")

target_column = 'prali_fire'

def preprocess_features(df):
    """Minimal feature engineering."""
    df = df.copy()
    df['acq_hour'] = df['acq_time'].astype(str).str.zfill(4).str[:2].astype(int)
    return df

print("Applying preprocessing...")
train_df_processed = preprocess_features(train_df)

#Define base feature sets
new_numeric_features = [
    'brightness', 'scan', 'track', 'confidence', 'bright_t31', 
    'frp', 'type', 'acq_hour'
]

new_categorical_features = [
    'satellite', 'instrument', 'daynight'
]

# Correlation Analysis
print("\nCorrelation Analysis...")
corr_matrix = train_df_processed[new_numeric_features + [target_column]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png")
print("Saved 'feature_correlation_heatmap.png'")

# Select features based on correlation...
print("\nSelecting Features Based on Correlation...")
corr_matrix_abs = corr_matrix.abs()
upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
threshold = 0.8
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
selected_numeric_features = [f for f in new_numeric_features if f not in to_drop]

print(f"Features removed due to high correlation (> {threshold}): {to_drop}")
print(f"Selected numeric features for training: {selected_numeric_features}")
print(f"Total numeric features reduced from {len(new_numeric_features)} → {len(selected_numeric_features)}")

# Combine numeric + categorical
all_features = selected_numeric_features + new_categorical_features

print(f"\nFinal feature count: {len(all_features)} (Numeric: {len(selected_numeric_features)}, Categorical: {len(new_categorical_features)})")

# Split data
X = train_df_processed[all_features]
y = train_df_processed[target_column]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Training Set: {X_train.shape}, Validation Set: {X_val.shape}, Test Set: {X_test.shape}")

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, selected_numeric_features),
        ('cat', categorical_transformer, new_categorical_features)
    ]
)

# Train multiple models
print("\nTraining Multiple Models...")
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = []
for name, clf in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    results.append([name, f"{acc:.4f}"])

print("\n--- Model Comparison Results ---")
print(tabulate(results, headers=["Model", "Validation Accuracy"], tablefmt="grid"))

# # train Only Fast Models ---
# print("\n training Fast Models Only")

# models = {
#     "Logistic Regression": LogisticRegression(max_iter=500),
#     "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
# }

# results = []
# for name, clf in models.items():
#     pipe = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', clf)
#     ])
#     pipe.fit(X_train, y_train)
#     y_pred = pipe.predict(X_val)
#     acc = accuracy_score(y_val, y_pred)
#     results.append([name, f"{acc:.4f}"])

# # Display results
# from tabulate import tabulate
# print("\n--- Model Comparison Results (Fast Models) ---")
# print(tabulate(results, headers=["Model", "Validation Accuracy"], tablefmt="grid"))


# Choose best model ---
best_model_name = max(results, key=lambda x: float(x[1]))[0]
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

pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', base_model)])

# Hyperparameter tuning
if param_grid:
    print("\nHyperparameter Tuning ")
    grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
else:
    best_pipeline = pipe.fit(X_train, y_train)

# Evaluate on Test Data
print("\n Final Evaluation")
y_pred_test = best_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_acc:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# Save model ---
joblib.dump(best_pipeline, 'prali_fire_model.pkl')
print("\nSaved best tuned model as 'prali_fire_model.pkl'")
