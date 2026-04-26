import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("face_mask_model.h5")

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
