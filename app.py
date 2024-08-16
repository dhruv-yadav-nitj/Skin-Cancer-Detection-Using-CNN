import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from keras.layers import Softmax

model = load_model('model.keras')

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
    confidence, pred_class_name = ans[0][pred_class], classes[pred_class]

    confidence_percentage = round(confidence * 100, 2)
    data = {
        'Predicted Class': [pred_class_name],
        'Confidence (%)': [confidence_percentage]
    }
    df = pd.DataFrame(data)
    st.write(df)
