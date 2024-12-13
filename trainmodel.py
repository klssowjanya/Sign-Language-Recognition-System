import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from function import actions, DATA_PATH, no_sequences, sequence_length

# Create a label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Load the data and create sequences
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            if os.path.exists(file_path):
                try:
                    res = np.load(file_path, allow_pickle=True)
                    if res.shape == (63,):  # Example shape check; adjust based on your data
                        window.append(res)
                    else:
                        print(f"Skipping file {file_path} due to unexpected shape: {res.shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"File {file_path} does not exist.")
        if len(window) == sequence_length:  # Ensure full window length
            sequences.append(window)
            labels.append(label_map[action])

# Debug: Print the shape of each sequence
for i, seq in enumerate(sequences):
    try:
        print(f"Sequence {i} shape: {np.array(seq).shape}")
    except Exception as e:
        print(f"Error with sequence {i}: {e}")

# Convert sequences and labels to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Set up TensorBoard callback for logging
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print the model summary
model.summary()

# Save the model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save('model.h5')