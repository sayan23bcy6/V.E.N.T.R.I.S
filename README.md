# V.E.N.T.R.I.S. - Vision-Enabled Gesture Recognition & Interaction System

This documentation outlines the functionality and setup of the **V.E.N.T.R.I.S.** script, a Python application that uses computer vision to recognize hand gestures in real-time and launch applications based on those gestures.

---

## Overview

- **Real-time gesture recognition** using your webcam.
- **Left-hand specific detection** to avoid accidental triggers.
- **Customizable actions**: Map gestures to any application via Windows shortcuts.
- **Action cooldown**: Prevents repetitive triggers.

The system uses a pre-trained machine learning model to classify hand gestures based on landmark data extracted by Google's MediaPipe library.

---

## Key Features

- **Real-time Gesture Recognition:** Uses the webcam for live video feed processing.
- **Hand-Specific Detection:** Exclusively processes gestures from the left hand, ignoring the right hand to prevent accidental activations.
- **Customizable Actions:** Allows users to map recognized gestures (letters) to any application or file via Windows shortcuts.
- **Action Cooldown:** Implements a delay after launching an app to prevent immediate, repetitive triggers of the same action.

---

## Dependencies

Install the following Python libraries:

```bash
pip install opencv-python numpy mediapipe joblib scikit-learn
```

- **OpenCV (cv2):** For capturing and handling the video feed from the webcam.
- **NumPy:** For efficient numerical operations, particularly for handling landmark coordinates.
- **MediaPipe:** For detecting and tracking hand landmarks in the video stream.
- **Joblib / Scikit-learn:** For loading the pre-trained machine learning model (`.pkl` file) and the associated label encoder.

---

## Setup and Configuration

### 1. Model File

Set the `MODEL_PATH` variable to your trained model file:

```python
MODEL_PATH = 'ventris_model_final.pkl'
```

This `.pkl` file should be a dictionary containing:
- `'model'`: The trained gesture classification model.
- `'label_encoder'`: The LabelEncoder object used to convert the model's numeric output back into a character label (e.g., `5 -> 'G'`).

---

### 2. Gesture-to-Application Mapping

The `MAPPINGS` dictionary links a recognized gesture (the key) to a specific Windows shortcut file (the value):

```python
MAPPINGS = {
    'A': 'AppName1.lnk',
    'B': 'AppName2.lnk',
    # Add other gestures and shortcuts here
}
```

**To set this up:**
1. Create a shortcut for the application you want to launch (right-click the `.exe` → Create shortcut).
2. Rename the shortcut file to match the value in the `MAPPINGS` dictionary (e.g., `AppName1.lnk`).
3. Place the shortcut file in the same directory as the Python script.

---

## How It Works

The script operates through a continuous loop, performing the following steps:

1. **Initialization:**
    - Loads the machine learning model and label encoder from `MODEL_PATH`.
    - Initializes MediaPipe's hand tracking solution.
    - Activates the webcam.

2. **Image Processing:**
    - Captures a frame from the webcam.
    - Flips the frame horizontally for a mirror-like view.
    - Converts the image from BGR to RGB for MediaPipe.

3. **Hand Landmark Detection:**
    - MediaPipe searches for hand landmarks.
    - Only left hand gestures are processed; right hand is ignored.

4. **Normalization and Prediction:**
    - Extracts 21 3D landmarks (x, y, z) for the left hand.
    - Normalizes landmarks relative to the wrist and scales them.
    - Feeds the flattened, normalized landmarks into the model.
    - Decodes the model's output into a gesture label (e.g., `'A'`, `'B'`).

5. **Action Triggering:**
    - If the gesture is mapped and different from the previous gesture, launches the corresponding shortcut using `os.startfile()`.
    - Pauses for `POST_LAUNCH_DELAY` seconds, displaying a countdown.

6. **Visual Feedback:**
    - Displays detection status (e.g., `DETECTED: G`, `RIGHT HAND DETECTED (IGNORED)`, `WAIT... 3`) on the video feed.

---

## Model Information

The classification model was trained on the `asl_hand_landmarks_multi.csv` dataset, which contains pre-processed hand landmark data for various American Sign Language (ASL) gestures.

---

## Usage

1. Ensure all dependencies are installed and configuration is complete.
2. Run the script from your terminal:

    ```bash
    python ventris_gui.py
    ```

3. A window titled **'V.E.N.T.R.I.S. - Live Feed'** will appear, showing your webcam feed.
4. Show a valid gesture with your left hand to the camera.
5. To stop the program, press the **'q'** key. The script will perform a clean shutdown, releasing the webcam and closing all windows.

---
