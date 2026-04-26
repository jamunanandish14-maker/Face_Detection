import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import urllib.request

MODEL_URL = "PASTE_KAGGLE_FILE_LINK_HERE"
MODEL_PATH = "face_mask_model.h5"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

st.title("Face Mask Detection")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file is not None:
    image = Image.open(file).resize((128,128))
    st.image(image)

    img = np.array(image)/255.0
    img = img.reshape(1,128,128,3)

    pred = model.predict(img)

    if pred[0][0] > 0.5:
        st.error("Without Mask")
    else:
        st.success("With Mask")
