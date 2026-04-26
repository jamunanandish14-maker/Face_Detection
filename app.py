import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))

st.title("Prediction App")

v1 = st.number_input("Feature 1")
v2 = st.number_input("Feature 2")

if st.button("Predict"):
    pred = model.predict([[v1,v2]])
    st.write(pred)
