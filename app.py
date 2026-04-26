import streamlit as st
import numpy as np

st.title("Prediction App")

v1 = st.number_input("Feature 1")
v2 = st.number_input("Feature 2")

# simple fallback prediction logic
def predict(v1, v2):
    return 1 if (v1 + v2) > 10 else 0

if st.button("Predict"):
    pred = predict(v1, v2)

    if pred == 1:
        st.success("Positive Class")
    else:
        st.error("Negative Class")
