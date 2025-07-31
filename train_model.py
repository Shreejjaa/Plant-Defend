
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Define dataset path and categories
DATASET_DIR = "dataset/"
CATEGORIES = ["healthy", "yellowing", "spotted", "dried"]
IMG_SIZE = 128

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.flatten()

features = []
labels = []

# Load dataset
for label, category in enumerate(CATEGORIES):
    path = os.path.join(DATASET_DIR, category)
    if not os.path.exists(path):
        continue
    for img_file in os.listdir(path):
        try:
            img_path = os.path.join(path, img_file)
            feat = extract_features(img_path)
            features.append(feat)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train classifier
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/leaf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully as 'model/leaf_model.pkl'")
