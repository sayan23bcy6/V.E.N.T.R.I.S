import cv2
import numpy as np
import mediapipe as mp
import joblib  # Using joblib is fine, it works like pickle
import os
import time

# --- MAPPINGS (Uses shortcut files in the same folder) ---
MAPPINGS = {
    'A': 'Anaconda Navigator.lnk',
    'B': 'Brave.lnk',
    'C': 'Command Prompt.lnk',
    'D': 'Discord.lnk',
    'G': 'Google Gemini.lnk',
    'J': 'Jupyter Notebook.lnk'
}

# --- CONFIGURATION ---
MODEL_PATH = 'ventris_model_final.pkl'
POST_LAUNCH_DELAY = 5  # 5-second delay after launching an app

# --- 1. LOAD THE MODEL AND THE LABEL ENCODER ---
print(f"Loading model from '{MODEL_PATH}'...")
try:
    # Load the dictionary that contains both the model and the encoder
    model_data = joblib.load(MODEL_PATH)
    
    # ** THE FIX IS HERE: Unpack the dictionary **
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    
    print("✅ Model and Label Encoder loaded successfully.")
except FileNotFoundError:
    print(f"--- ERROR: Model file not found ('{MODEL_PATH}') ---")
    exit()
except KeyError:
    print(f"--- ERROR: Model file is not in the correct format. ---")
    print("It must be a dictionary with 'model' and 'label_encoder' keys.")
    exit()

# --- MEDIAPIPE & NORMALIZATION SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def normalize_landmarks(landmarks_list):
    landmarks = np.array(landmarks_list).reshape(-1, 3)
    if np.all(landmarks == 0): return landmarks.flatten()
    
    # Use the same normalization as the training script
    # 1. Translation
    origin = landmarks[0].copy()
    landmarks_translated = landmarks - origin
    
    # 2. Scaling
    max_val = np.abs(landmarks_translated).max()
    if max_val == 0: return landmarks_translated.flatten()
    
    normalized_landmarks = landmarks_translated / max_val
    return normalized_landmarks.flatten()

# --- GESTURE RECOGNITION LOOP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("--- ERROR: Could not open webcam. ---")
    exit()

print("\n--- V.E.N.T.R.I.S. Activated ---")
print("Show a gesture from your LEFT hand. Press 'q' to quit.")

last_gesture = None
display_text = "Awaiting input..."

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue

    frame_flipped = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    current_prediction = "nothing"

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_info = results.multi_handedness[0].classification[0]
        hand_label = hand_info.label

        if hand_label == "Left":
            landmarks = results.multi_hand_landmarks[0].landmark
            landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]
            normalized_landmarks = normalize_landmarks(landmarks_list)
            
            # ** THE FIX IS HERE: Predict and Decode the result **
            # 1. Predict the numeric label (e.g., 5)
            prediction_encoded = model.predict([normalized_landmarks])
            # 2. Decode the numeric label into text (e.g., 'G') using the encoder
            current_prediction = label_encoder.inverse_transform(prediction_encoded)[0]
            
            display_text = f"DETECTED: {current_prediction.upper()}"
        else:
            display_text = "RIGHT HAND DETECTED (IGNORED)"
            current_prediction = "nothing"
    else:
        display_text = "DETECTED: NOTHING"
        current_prediction = "nothing"

    # Display prediction on the screen
    cv2.putText(frame_flipped, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # --- ACTION TRIGGER LOGIC ---
    if current_prediction in MAPPINGS and current_prediction != last_gesture:
        app_shortcut = MAPPINGS[current_prediction]
        if os.path.exists(app_shortcut):
            print(f"ACTION: Detected '{current_prediction}'. Launching '{app_shortcut}'...")
            try:
                os.startfile(app_shortcut)
                print(f"Success! Pausing for {POST_LAUNCH_DELAY} seconds...")
                # Display a countdown on the screen
                for i in range(POST_LAUNCH_DELAY, 0, -1):
                    ret_wait, frame_wait = cap.read()
                    if not ret_wait: break
                    frame_wait_flipped = cv2.flip(frame_wait, 1)
                    # Draw the original detection text
                    cv2.putText(frame_wait_flipped, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    # Draw the countdown text
                    cv2.putText(frame_wait_flipped, f"WAIT... {i}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('V.E.N.T.R.I.S. - Live Feed', frame_wait_flipped)
                    cv2.waitKey(1000) # Wait for 1 second
            except Exception as e:
                print(f"Error launching '{app_shortcut}': {e}")
        else:
            print(f"Error: Shortcut '{app_shortcut}' not found.")
            # Display error on screen for a moment
            cv2.putText(frame_flipped, f"ERROR: Shortcut for {current_prediction} not found", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow('V.E.N.T.R.I.S. - Live Feed', frame_flipped)
            cv2.waitKey(2000)

    # Update the last seen gesture to prevent spamming actions
    if current_prediction != "nothing":
        last_gesture = current_prediction
    else:
        # Allow a new gesture to be triggered immediately after the hand disappears
        last_gesture = None
    
    cv2.imshow('V.E.N.T.R.I.S. - Live Feed', frame_flipped)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
hands.close()
cap.release()
cv2.destroyAllWindows()
print("V.E.N.T.R.I.S. has been shut down.")
