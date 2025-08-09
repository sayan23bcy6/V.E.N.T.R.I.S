V.E.N.T.R.I.S. - Vision-Enabled Gesture Recognition & Interaction System
This documentation outlines the functionality and setup of the V.E.N.T.R.I.S. script, a Python application that uses computer vision to recognize hand gestures in real-time and launch applications based on those gestures.

Overview
The script activates a webcam to monitor for hand gestures. It is specifically designed to recognize gestures made by the left hand. Upon identifying a pre-configured gesture, it launches a corresponding application by executing a Windows shortcut (.lnk) file. The system uses a pre-trained machine learning model to classify the hand gestures based on landmark data extracted by Google's MediaPipe library.

Key Features:

Real-time Gesture Recognition: Uses the webcam for live video feed processing.

Hand-Specific Detection: Exclusively processes gestures from the left hand, ignoring the right hand to prevent accidental activations.

Customizable Actions: Allows users to map recognized gestures (letters) to any application or file via Windows shortcuts.

Action Cooldown: Implements a delay after launching an app to prevent immediate, repetitive triggers of the same action.

Dependencies
The script requires the following Python libraries. You can install them using pip:

Bash

pip install opencv-python numpy mediapipe joblib scikit-learn
OpenCV (cv2): For capturing and handling the video feed from the webcam.

NumPy: For efficient numerical operations, particularly for handling landmark coordinates.

MediaPipe: For detecting and tracking hand landmarks in the video stream.

Joblib / Scikit-learn: For loading the pre-trained machine learning model (.pkl file) and the associated label encoder.

Setup and Configuration
Before running the script, you need to configure a few components located at the top of the file.

1. Model File
The MODEL_PATH variable must point to the trained model file.

Python

MODEL_PATH = 'ventris_model_final.pkl'
This .pkl file is expected to be a dictionary containing two objects:

'model': The trained gesture classification model.

'label_encoder': The LabelEncoder object used to convert the model's numeric output back into a character label (e.g., 5 -> 'G').

2. Gesture-to-Application Mapping
The MAPPINGS dictionary is the core of the script's functionality. It links a recognized gesture (the key) to a specific Windows shortcut file (the value).

Python

MAPPINGS = {
    'A': 'AppName1.lnk',
    'B': 'AppName2.lnk',
    # Add other gestures and shortcuts here
}
To make this work:

Create a shortcut for the application you want to launch (e.g., right-click the .exe -> Create shortcut).

Rename the shortcut file to match the value in the MAPPINGS dictionary (e.g., AppName1.lnk).

Place the shortcut file in the same directory as the Python script.

How It Works
The script operates through a continuous loop, performing the following steps:

Initialization:

The machine learning model and label encoder are loaded from the file specified by MODEL_PATH.

MediaPipe's hand tracking solution is initialized to detect a single hand with a high confidence threshold.

The webcam is activated.

Image Processing:

A frame is captured from the webcam.

The frame is flipped horizontally (cv2.flip) to create a more intuitive, mirror-like view for the user.

The image's color space is converted from BGR (OpenCV's default) to RGB, which is the format required by MediaPipe.

Hand Landmark Detection:

The RGB image is processed by MediaPipe, which searches for hand landmarks.

If a hand is detected, the script checks if it's the left hand. Right-hand detections are explicitly ignored.

Normalization and Prediction:

For a detected left hand, the 21 3D landmarks (x,y,z) are extracted.

These landmarks undergo normalization. The normalize_landmarks function translates all points relative to the wrist (landmark 0) and scales them. This crucial step ensures that the model's prediction is not affected by the hand's size or distance from the camera.

The flattened array of normalized landmarks is fed into the loaded machine learning model.

The model predicts a numeric label, which is then decoded back into its character form (e.g., 'A', 'B') using the label_encoder.

Action Triggering:

The script checks if the predicted gesture is a key in the MAPPINGS dictionary and if it's different from the previously detected gesture (to avoid spam).

If both conditions are met, it uses os.startfile() to execute the corresponding shortcut file, effectively launching the application.

After launching an app, the script pauses for the duration of POST_LAUNCH_DELAY, displaying a countdown on the screen to inform the user.

Visual Feedback:

The current detection status (e.g., DETECTED: G, RIGHT HAND DETECTED (IGNORED), WAIT... 3) is drawn directly onto the video feed shown to the user.

Model Information
The classification model was trained on the asl_hand_landmarks_multi.csv dataset. This dataset contains pre-processed hand landmark data for various American Sign Language (ASL) gestures, making the model effective at distinguishing between different hand shapes.

Usage
Ensure all dependencies are installed and the configuration is complete.

Run the script from your terminal:

Bash

python ventris_gui.py
A window titled 'V.E.N.T.R.I.S. - Live Feed' will appear, showing your webcam feed.

Show a valid gesture with your left hand to the camera.

To stop the program, press the 'q' key. The script will then perform a clean shutdown, releasing the webcam and closing all windows.