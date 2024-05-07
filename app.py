import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from keras.layers import Softmax

model = load_model('model.h5')

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
    

    pred_class = np.argmax(ans)
    confidence, pred_class_name = ans[pred_class]*100, classes[pred_class]

    st.write(f"Predicted Class: {pred_class_name}, Confidence: {confidence * 100:.2f}%")