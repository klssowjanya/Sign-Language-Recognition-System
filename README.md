# Sign Language Recognition

## Overview
The Sign Language Recognition project is designed to identify and interpret hand gestures corresponding to letters of the alphabet. Using computer vision and machine learning techniques, the project aims to bridge the communication gap between hearing and non-hearing individuals by translating sign language into readable text.

## Features
- *Real-time Hand Gesture Detection*: Captures hand gestures using a webcam.
- *Alphabet Recognition*: Identifies and translates gestures for individual letters of the alphabet.
- *Dataset Creation*: Collects images of hand gestures for training the model.
- *Model Training*: Trains a machine learning model on the captured dataset to improve recognition accuracy.

## Technology Stack
- *Programming Language*: Python
- *Libraries and Frameworks*:
  - OpenCV (for image processing and webcam integration)
  - MediaPipe (for hand landmark detection)
  - TensorFlow/Keras (for model training and prediction)
  - NumPy, Pandas (for data handling)
- *Model Architecture*: CNN-LSTM (for feature extraction and sequential data processing)
- *Hardware Requirements*: A webcam-enabled computer

## Installation
1. Clone this repository:
   bash
   git clone <[repository-url](https://github.com/klssowjanya/Sign-Language-Recognition-System)>
   
2. Navigate to the project directory:
   bash
   cd sign-language-recognition
   
3. Install required dependencies:
   bash
   pip install -r requirements.txt
   

## Dataset Collection
1. Run the collectdata.py script to collect hand gesture images:
   bash
   python collectdata.py
   
2. Press the alphabet key (e.g., A, B) to capture images for the respective gesture.
3. Images will be saved in their respective alphabet folders for training purposes.

![Screenshot ![Screenshot 2024-12-12 193356](https://github.com/user-attachments/assets/c2ec5cee-863f-4bf3-a145-1405faf2019b)



## Model Training
1. Prepare the dataset by ensuring each folder contains sufficient images for each letter.
2. Run the training script:
   bash
   python trainmodel.py
   
3. The trained model will be saved as model.h5 in the project directory.

![Screenshot 2024-12-12 193132](https://github.com/user-attachments/assets/c7010daf-26c6-4420-954c-cf8381a42ea5)


## Usage
1. Start the recognition script:
   bash
   python app.py
   
2. Show hand gestures in front of the webcam.
3. The recognized letter will be displayed on the screen.

![Screenshot 2024-12-12 193027](https://github.com/user-attachments/assets/a71e01d0-767d-4363-8f38-d3b17d690702)




## Project Workflow
1. *Data Collection*: Capture images for each sign.
2. *Preprocessing*: Resize, normalize, and augment images to enhance model performance.
3. *Model Training*: Train a Convolutional Neural Network (CNN) with LSTM for sequential feature learning.
4. *Real-Time Recognition*: Use the trained model along with MediaPipe for predicting gestures in real-time.

## Future Enhancements
- Add support for recognizing sign language words and sentences.
- Improve model accuracy with more diverse datasets.
- Implement a GUI for easier interaction.
- Integrate with speech synthesis to voice out recognized letters or words.

## Acknowledgements
- OpenCV documentation and tutorials
- MediaPipe for hand tracking support
- TensorFlow/Keras for deep learning framework
- Faculty and peers for their guidance

---
*Note*: This project was developed as part of an academic initiative to explore computer vision and machine learning applications.
