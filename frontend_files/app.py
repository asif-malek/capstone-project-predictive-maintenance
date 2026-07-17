import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
# Ensure this filename matches what was logged by MLflow and uploaded to Hugging Face
model_path = hf_hub_download(repo_id="asifaddicted/vehicle-breakdown-prediction-model", filename="adaboost_classifier_model.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Vehicle Breakdown Prediction")
st.write("""
This application predicts whether an engine is at **risk of breakdown (Class 1)** or is
operating **normally (Class 0)** based on real-time sensor readings.
Please enter the engine parameters below to get a prediction.
""")

# User input for the original engine parameters
engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=790, step=10)
lub_oil_pressure = st.number_input("Lubrication Oil Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=3.3, step=0.1)
fuel_pressure = st.number_input("Fuel Pressure (bar/kPa)", min_value=0.0, max_value=30.0, value=6.6, step=0.1)
coolant_pressure = st.number_input("Coolant Pressure (bar/kPa)", min_value=0.0, max_value=10.0, value=2.3, step=0.1)
lub_oil_temp = st.number_input("Lubrication Oil Temperature (°C)", min_value=0.0, max_value=150.0, value=77.6, step=0.1)
coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=250.0, value=78.4, step=0.1)

# Predict button
if st.button("Predict Engine Condition"):
    # Create a DataFrame from user inputs
    input_data_df = pd.DataFrame([{
        'Engine_rpm': engine_rpm,
        'Lub_oil_pressure': lub_oil_pressure,
        'Fuel_pressure': fuel_pressure,
        'Coolant_pressure': coolant_pressure,
        'lub_oil_temp': lub_oil_temp,
        'Coolant_temp': coolant_temp
    }])

    # Apply the same feature engineering as in prep.py and the notebook
    input_data_df["RPM_to_OilPres_Ratio"] = input_data_df["Engine_rpm"] / (input_data_df["Lub_oil_pressure"] + 0.001)
    input_data_df["Temp_Divergence"] = abs(input_data_df["lub_oil_temp"] - input_data_df["Coolant_temp"])
    input_data_df["Total_Pressure"] = input_data_df["Lub_oil_pressure"] + input_data_df["Fuel_pressure"] + input_data_df["Coolant_pressure"]

    # Get probabilities for the positive class (class 1)
    prediction_proba = model.predict_proba(input_data_df)[0][1] # Probability of class 1

    # Apply the 0.30 threshold for aggressive monitoring
    prediction = (prediction_proba >= 0.30).astype(int)

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"Engine Condition: **Breakdown Risk (Class 1)** with probability {prediction_proba:.2f} (using 0.30 threshold)")
    else:
        st.success(f"Engine Condition: **Normal Operation (Class 0)** with probability {1 - prediction_proba:.2f} (using 0.30 threshold)")

    st.write("Note: Class 1 indicates a high likelihood of breakdown or a condition requiring maintenance.")
