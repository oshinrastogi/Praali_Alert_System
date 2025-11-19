import streamlit as st
import pandas as pd
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Prali Fire Prediction",
    page_icon="🔥",
    layout="wide"
)

# Model Loading 
MODEL_FILE = 'prali_fire_model.pkl'

@st.cache_resource
def load_model(model_path):
    """Loads the saved model pipeline from a file."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at '{model_path}'")
        st.error("Please make sure 'prali_fire_model.pkl' is in the same directory as 'app.py'")
        return None
    
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model_pipeline = load_model(MODEL_FILE)

# Feature Definitions 
NEW_NUMERIC_FEATURES = [
    'brightness', 'scan', 'track', 'confidence', 'bright_t31', 
    'frp', 'type', 'acq_hour'
]

NEW_CATEGORICAL_FEATURES = [
    'satellite', 'instrument', 'daynight'
]

ALL_FEATURES = NEW_NUMERIC_FEATURES + NEW_CATEGORICAL_FEATURES

# --- App Layout ---
st.title("🔥 Prali Fire (Stubble Burning) Prediction")
st.markdown("This app uses a Random Forest model to predict the likelihood of a fire being a 'Prali Fire' based on its physical properties.")

if model_pipeline is None:
    st.stop()

# --- Sidebar for User Input ---
st.sidebar.header("Input Fire Characteristics")

# Create a dictionary to hold user inputs
inputs = {}

# --- Numeric Inputs ---
st.sidebar.subheader("Numeric Features")
inputs['brightness'] = st.sidebar.number_input("Brightness (Kelvin)", min_value=250.0, max_value=500.0, value=330.0, step=0.1)
inputs['bright_t31'] = st.sidebar.number_input("Brightness (T31 Band, Kelvin)", min_value=250.0, max_value=400.0, value=300.0, step=0.1)
inputs['frp'] = st.sidebar.number_input("Fire Radiative Power (FRP)", min_value=0.0, max_value=500.0, value=25.0, step=0.1)
inputs['scan'] = st.sidebar.slider("Scan Size", min_value=0.1, max_value=2.0, value=0.5, step=0.01)
inputs['track'] = st.sidebar.slider("Track Size", min_value=0.1, max_value=2.0, value=0.4, step=0.01)
inputs['confidence'] = st.sidebar.slider("Confidence (0-100)", min_value=0, max_value=100, value=80)
inputs['type'] = st.sidebar.selectbox("Fire Type (0=Presumed, 2=Other)", [0, 1, 2, 3], index=0)
inputs['acq_hour'] = st.sidebar.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=14)

# --- Categorical Inputs ---
st.sidebar.subheader("Categorical Features")
inputs['satellite'] = st.sidebar.selectbox("Satellite", ['N20', 'Aqua', 'Terra']) 
inputs['instrument'] = st.sidebar.selectbox("Instrument", ['VIIRS', 'MODIS'])
inputs['daynight'] = st.sidebar.selectbox("Day or Night", ['D', 'N'])


# --- Main Page for Prediction ---
st.header("Model Prediction")

if st.button("Predict Likelihood", type="primary"):

    input_df = pd.DataFrame([inputs])
    
    try:
        input_df = input_df[ALL_FEATURES]
    except KeyError as e:
        st.error(f"Feature mismatch error: {e}. Ensure all features are provided.")
        st.stop()

    st.subheader("Input Data:")
    st.dataframe(input_df)

    try:
        # Predict class (0 or 1)
        prediction = model_pipeline.predict(input_df)
        
        # Predict probabilities [prob_of_0, prob_of_1]
        probabilities = model_pipeline.predict_proba(input_df)
        
        prali_probability = probabilities[0][1] # Probability of class 1
        predicted_class = prediction[0]

        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)

        with col1:
            if predicted_class == 1:
                st.error("Prediction: LIKELY a Prali Fire")
            else:
                st.success("Prediction: NOT likely a Prali Fire")

        with col2:
            st.metric(
                label="Probability of Prali Fire",
                value=f"{prali_probability * 100:.2f} %"
            )
        
        st.progress(prali_probability)

        st.info("This model is based on the *physical properties* of the fire, not its location or date.")


    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")