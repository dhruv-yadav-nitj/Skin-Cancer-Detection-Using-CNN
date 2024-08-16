# Skin Cancer Detection Using CNN

This project aims to detect melanoma, a type of skin cancer, using Convolutional Neural Networks (CNN). The model is trained on the HAM10000 dataset, which contains images of localized skin cells. It predicts whether a tumor in the input image is benign or malignant with an accuracy of approximately 90%.

## Dataset
The HAM10000 dataset consists of 10000 dermatoscopic images. Each image is labeled with either 'Benign' or 'Malignant' class. If the given input image lies in 'Benign' class then the cells are non-cancerous else they are cancerous.

## Technologies Used
- TensorFlow
- Keras
- Streamlit

## Model Architecture
The Convolutional Neural Network (CNN) architecture used for this project consists of multiple convolutional layers followed by max-pooling layers for feature extraction. The extracted features are then passed through fully connected layers for classification. The model is trained using the HAM10000 dataset with appropriate data augmentation techniques to improve generalization.

## Confusion Matrix
![confusion-matrix](https://github.com/user-attachments/assets/c8c35983-333a-42c7-b6f6-dec633a66bdc)

## Deployment
The project is deployed using Streamlit here: [Live](https://karanveer-detects.streamlit.app/)
