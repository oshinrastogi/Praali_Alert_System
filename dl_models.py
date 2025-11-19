import os
import random
import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate

# Reproducibility
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Config
DATA_FILE = "training_data.csv"
TARGET_COL = "prali_fire"
CORR_THRESHOLD = 0.80
EPOCHS = 30
BATCH_SIZE = 32
PATIENCE = 4
SAVE_MODELS = True  

# Load
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in cwd.")

df = pd.read_csv(DATA_FILE)
print(f"Loaded {DATA_FILE} — shape: {df.shape}")

if 'acq_time' in df.columns:
    df['acq_hour'] = df['acq_time'].astype(str).str.zfill(4).str[:2].astype(int)
else:
    if 'acq_hour' not in df.columns:
        raise ValueError("Neither 'acq_time' nor 'acq_hour' present in dataset. Feature engineering expects one of them.")

numeric_features = [
    'brightness', 'scan', 'track', 'confidence', 'bright_t31', 
    'frp', 'type', 'acq_hour', 'NDVI', 'NBR', 'NDWI','acq_time',
]

categorical_features = ['satellite', 'instrument', 'daynight']

missing_numeric = [c for c in numeric_features if c not in df.columns]
missing_categorical = [c for c in categorical_features if c not in df.columns]
if missing_numeric:
    raise ValueError(f"Missing numeric columns: {missing_numeric}")
if missing_categorical:
    raise ValueError(f"Missing categorical columns: {missing_categorical}")
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found")

df = df.dropna(subset=numeric_features + categorical_features + [TARGET_COL]).reset_index(drop=True)
print(f"After dropping NA rows: {df.shape}")

# Correlation-based numeric feature selection
corr_df = df[numeric_features + [TARGET_COL]].corr().abs()
upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
selected_numeric = [c for c in numeric_features if c not in to_drop]

print("\nCorrelation analysis:")
print(f" - Dropping numeric features with corr > {CORR_THRESHOLD}: {to_drop}")
print(f" - Selected numeric features: {selected_numeric}")

selected_features = selected_numeric + categorical_features
print(f"Final selected features (count={len(selected_features)}): {selected_features}")

# Save selected features
joblib.dump(selected_features, "lstm_selected_features.pkl")
print("Saved selected features → lstm_selected_features.pkl")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), selected_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)

X = df[selected_features]
y = df[TARGET_COL].astype(int).values

X_transformed = preprocessor.fit_transform(X)  
print(f"Transformed feature shape: {X_transformed.shape}")

joblib.dump(preprocessor, "dl_preprocessor.pkl")
print("Saved preprocessor → dl_preprocessor.pkl")

#  Train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_transformed, y, test_size=0.30, stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, class_weights)}
print(f"Computed class weights: {class_weight_dict}")

# 5. Prepare shapes for models
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_rnn   = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_rnn  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# 1D-CNN
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_cnn   = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_cnn  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

input_dim = X_train.shape[1]
print(f"Input dim (num features after preprocessing): {input_dim}")

# Model builder helpers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def compile_and_train(model, X_tr, y_tr, X_val_, y_val_, model_name):
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
    rr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0)
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val_, y_val_),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rr],
        class_weight=class_weight_dict,
        verbose=1
    )
    if SAVE_MODELS:
        fname = f"{model_name}.keras"
        if os.path.exists(fname):
            os.remove(fname)
        model.save(fname)
        print(f"Saved model => {fname}")

    return model, history

# Define & train models
trained_models = []
metrics = []

# MLP 
print("\n--- TRAINING MLP (Dense NN) ---")
mlp = Sequential([
    Dense(128, activation="relu", input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(len(classes), activation="softmax")
])
mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
mlp, _ = compile_and_train(mlp, X_train, y_train, X_val, y_val, "mlp_model")
trained_models.append(("MLP", mlp))

# GRU
print("\n--- TRAINING GRU ---")
gru = Sequential([
    GRU(64, activation="tanh", input_shape=(1, input_dim)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(classes), activation="softmax")
])
gru.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
gru, _ = compile_and_train(gru, X_train_rnn, y_train, X_val_rnn, y_val, "gru_model")
trained_models.append(("GRU", gru))

#  LSTM
print("\n--- TRAINING LSTM ---")
lstm = Sequential([
    LSTM(64, activation="tanh", input_shape=(1, input_dim)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(classes), activation="softmax")
])
lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
lstm, _ = compile_and_train(lstm, X_train_rnn, y_train, X_val_rnn, y_val, "lstm_model")
trained_models.append(("LSTM", lstm))

# BiLSTM
print("\n--- TRAINING BiLSTM ---")
bilstm = Sequential([
    Bidirectional(LSTM(64, activation="tanh"), input_shape=(1, input_dim)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(len(classes), activation="softmax")
])
bilstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
bilstm, _ = compile_and_train(bilstm, X_train_rnn, y_train, X_val_rnn, y_val, "bilstm_model")
trained_models.append(("BiLSTM", bilstm))

# 1D-CNN
print("\n--- TRAINING 1D-CNN ---")
cnn = Sequential([
    Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=(input_dim, 1)),
    Conv1D(filters=32, kernel_size=3, activation="relu"),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(classes), activation="softmax")
])
cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn, _ = compile_and_train(cnn, X_train_cnn, y_train, X_val_cnn, y_val, "cnn1d_model")
trained_models.append(("1D-CNN", cnn))

# Evaluate all models on test set
print("\n--- EVALUATION ON TEST SET ---")
results = []
for name, model in trained_models:
    if name == "MLP":
        X_in = X_test
    elif name == "1D-CNN":
        X_in = X_test_cnn
    else:
        # GRU, LSTM, BiLSTM
        X_in = X_test_rnn

    y_prob = model.predict(X_in, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

    print(f"\n{name} classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# Print comparison table
df_results = pd.DataFrame(results).sort_values("accuracy", ascending=False).reset_index(drop=True)
print("\n=== MODEL COMPARISON ===")
print(tabulate(df_results, headers="keys", tablefmt="github", showindex=True, floatfmt=".4f"))

# save comparison
df_results.to_csv("dl_models_comparison.csv", index=False)
print("\nSaved comparison → dl_models_comparison.csv")

print("\nAll done.")
