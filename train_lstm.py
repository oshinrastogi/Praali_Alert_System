import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ----------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------
print("Loading training data...")

if not os.path.exists("training_data.csv"):
    print("Error: training_data.csv not found.")
    exit()

df = pd.read_csv("training_data.csv")
print(f"Loaded training_data.csv (Shape: {df.shape})")

target = "prali_fire"


# ----------------------------------------------------
# 2. FEATURE ENGINEERING (same as your main file)
# ----------------------------------------------------
def preprocess_features(df):
    df = df.copy()
    df['acq_hour'] = df['acq_time'].astype(str).str.zfill(4).str[:2].astype(int)
    return df

df = preprocess_features(df)

numeric_features = [
    'brightness', 'scan', 'track', 'confidence', 'bright_t31',
    'frp', 'type', 'acq_hour'
]

categorical_features = [
    'satellite', 'instrument', 'daynight'
]


# ----------------------------------------------------
# 3. CORRELATION-BASED FEATURE SELECTION
# ----------------------------------------------------
print("\nComputing correlation matrix...")

corr_matrix = df[numeric_features + [target]].corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.80
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

selected_numeric_features = [f for f in numeric_features if f not in to_drop]

print("\n🔥 FEATURES REMOVED (corr > 0.80):", to_drop)
print("🔥 SELECTED NUMERIC FEATURES:", selected_numeric_features)


# Final feature list
selected_features = selected_numeric_features + categorical_features
print("\nFinal feature count:", len(selected_features))


# ----------------------------------------------------
# 4. DROP NA & PREPARE DATA
# ----------------------------------------------------
df = df.dropna(subset=selected_features + [target])
X = df[selected_features]
y = df[target]


# ----------------------------------------------------
# 5. PREPROCESSING PIPELINE (SCALER + ONEHOT)
# ----------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), selected_numeric_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_transformed = preprocessor.fit_transform(X)

joblib.dump(preprocessor, "lstm_preprocessor.pkl")
joblib.dump(selected_features, "lstm_selected_features.pkl")
print("\nSaved selected features → lstm_selected_features.pkl")


# ----------------------------------------------------
# 6. SPLIT DATA
# ----------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_transformed, y, test_size=0.30, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)


# ----------------------------------------------------
# 7. RESHAPE FOR LSTM (samples, time_step=1, features)
# ----------------------------------------------------
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_val_lstm   = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
X_test_lstm  = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
y_val_cat   = tf.keras.utils.to_categorical(y_val, 2)
y_test_cat  = tf.keras.utils.to_categorical(y_test, 2)

input_dim = X_train_lstm.shape[2]


# ----------------------------------------------------
# 8. BUILD LSTM MODEL
# ----------------------------------------------------
print("\nBuilding LSTM model...")

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(1, input_dim)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ----------------------------------------------------
# 9. TRAIN MODEL
# ----------------------------------------------------
print("\nTraining LSTM...")

early_stop = EarlyStopping(
    monitor='val_loss', patience=4, restore_best_weights=True
)

history = model.fit(
    X_train_lstm, y_train_cat,
    validation_data=(X_val_lstm, y_val_cat),
    epochs=25,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# ----------------------------------------------------
# 10. EVALUATE
# ----------------------------------------------------
loss, accuracy = model.evaluate(X_test_lstm, y_test_cat, verbose=0)
print(f"\n🔥 LSTM Test Accuracy: {accuracy:.4f}")


# ----------------------------------------------------
# 11. SAVE MODEL
# ----------------------------------------------------
model.save("prali_fire_lstm.h5")
print("\nModel saved → prali_fire_lstm.h5")
print("Preprocessor saved → lstm_preprocessor.pkl")
print("Selected features saved → lstm_selected_features.pkl")
