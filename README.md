readme_content = """
# Skin Cancer Detection Using CNN

This project aims to detect melanoma, a type of skin cancer, using Convolutional Neural Networks (CNN). The model is trained on the HAM10000 dataset, which contains images of localized skin cells. It predicts whether a tumor in the input image is benign or malignant with an accuracy of approximately 90%.

## Dataset
The HAM10000 dataset consists of 10000 dermatoscopic images. Each image is labeled with one of seven diagnostic categories: melanoma, melanocytic nevus, basal cell carcinoma, actinic keratosis, benign keratosis, dermatofibroma, or vascular lesion. For this project, we focused on detecting melanoma.

## Technologies Used
- TensorFlow
- Keras
- Streamlit

## Model Architecture
The Convolutional Neural Network (CNN) architecture used for this project consists of multiple convolutional layers followed by max-pooling layers for feature extraction. The extracted features are then passed through fully connected layers for classification. The model is trained using the HAM10000 dataset with appropriate data augmentation techniques to improve generalization.

[Deployment](karanveer-detects.streamlit.app)
