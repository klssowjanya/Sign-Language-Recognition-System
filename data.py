from function import *
import os
import cv2
import numpy as np
import mediapipe as mp

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')

# Actions to detect
actions = np.array(['A','B','C','D','E','F','G','H','I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W','X','Y','Z'])

# Number of sequences and sequence length
no_sequences = 50
sequence_length = 50

# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences (videos)
        for sequence in range(no_sequences):
            # Loop through video length (sequence length)
            for frame_num in range(sequence_length):

                # Construct the image path
                image_path = f'Image/{action}/{sequence}.png'
                frame = cv2.imread(image_path)
                
                # Check if the image was loaded successfully
                if frame is None:
                    print(f"Error: Image not found or unable to load: {image_path}")
                    continue  # Skip this frame and move to the next one

                # Perform MediaPipe detection
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cv2.destroyAllWindows()
