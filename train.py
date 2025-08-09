import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Dataset ---
print("Loading dataset...")
try:
    df = pd.read_csv('asl_hand_landmarks_multi.csv')
    print(f"Dataset loaded successfully with {len(df)} samples.")
    print(f"Classes found: {df['label'].unique()}")
except FileNotFoundError:
    print("Error: 'asl_hand_landmarks_multi.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

# --- 2. Preprocess the Data ---

# Separate features (X) and labels (y)
X = df.drop('label', axis=1)
y = df['label']

# Encode string labels into numbers
print("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Labels encoded into {len(np.unique(y_encoded))} numeric classes.")

# Normalize the landmark data
print("Normalizing landmark data...")
def normalize_landmarks(data):
    normalized_data = []
    for index, row in data.iterrows():
        # Reshape the flat row into 21 landmarks with (x, y, z) coordinates
        landmarks = np.array(row).reshape(-1, 3)

        # 1. Translation: Move all points relative to the wrist (landmark 0)
        base = landmarks[0].copy()
        translated_landmarks = landmarks - base

        # 2. Normalization: Scale the landmarks to be size-invariant
        # Find the maximum absolute value to scale all coordinates
        max_val = np.abs(translated_landmarks).max()
        if max_val > 0:
            normalized_landmarks = translated_landmarks / max_val
        else:
            normalized_landmarks = translated_landmarks # Avoid division by zero

        # Flatten the normalized landmarks back into a single row
        normalized_data.append(normalized_landmarks.flatten())

    return np.array(normalized_data)

X_normalized = normalize_landmarks(X)
print("Normalization complete.")

# --- 3. Split Data for Training and Testing ---
print("Splitting data into training and testing sets...")
# Use stratify to ensure the class distribution is the same in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized,
    y_encoded,
    test_size=0.2,  # 80% for training, 20% for testing
    random_state=42,
    stratify=y_encoded
)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")


# --- 4. Train the Machine Learning Model ---
print("Training the RandomForestClassifier model...")
# RandomForest is a strong choice for this type of structured data
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete. ✨")


# --- 5. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Optional: Display a confusion matrix to see performance per class
print("Generating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'.")


# --- 6. Save the Model and Encoder ---
model_filename = 'ventris_model.pkl'
print(f"\nSaving model and label encoder to '{model_filename}'...")

# Save both the model and the label encoder in a dictionary
model_data = {
    'model': model,
    'label_encoder': label_encoder
}

with open(model_filename, 'wb') as f:
    pickle.dump(model_data, f)

print(f"🎉 Model saved successfully! You can now use '{model_filename}' for predictions.")