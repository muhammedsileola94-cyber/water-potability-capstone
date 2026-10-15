import joblib
import pandas as pd
import streamlit as st

model = joblib.load("water_model.joblib")

st.title("Water Potability Predictor")
st.write("Enter water quality measurements to predict whether the water is potable.")

features = {
    "ph": 7.0, "Hardness": 196.0, "Solids": 22000.0, "Chloramines": 7.1,
    "Sulfate": 333.0, "Conductivity": 426.0, "Organic_carbon": 14.3,
    "Trihalomethanes": 66.0, "Turbidity": 4.0,
}

values = {name: st.number_input(name, value=default) for name, default in features.items()}

if st.button("Predict"):
    X = pd.DataFrame([values], columns=list(features))
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    if pred == 1:
        st.success(f"Predicted potable (probability {prob:.2f})")
    else:
        st.error(f"Predicted not potable (probability of potable {prob:.2f})")
    st.caption("Model accuracy on test data is about 67%. Do not use this to judge real drinking water.")