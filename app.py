import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from keras.layers import Softmax

# Define custom objects with Softmax activation function
custom_objects = {'Softmax': Softmax}

# Load the model with custom objects
model = load_model('modelv1.h5', custom_objects=custom_objects)


st.title("Skin Cancer Prediction System")

uploaded = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Input Cell Image", width=200)

    def preprocess_image(uploaded_image):
        resized_image = uploaded_image.resize((256, 256))
        image_array = img_to_array(resized_image)
        image_array /= 255.
        return image_array

    def prediction(image_array):
        pred = model.predict(np.expand_dims(image_array, axis=0))
        return pred

    inp = preprocess_image(image)
    ans = prediction(inp)
    classes = ['Benign', 'Malignant']
    st.write(ans); st.write (classes[np.argmax(ans)])