# Import dependencies
import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image.flags.writeable = False  # Mark image as not writeable for efficiency
    results = model.process(image)  # Process the image and make predictions
    image.flags.writeable = True  # Mark image as writeable again
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR
    return image, results

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results): 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract keypoints
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            return np.concatenate([rh])
    # Return a zero array if no landmarks are found
    

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions to detect
actions = np.array(['A','B','C','D','E','F','G','H','I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# Number of sequences and sequence length
no_sequences = 50
sequence_length = 50
