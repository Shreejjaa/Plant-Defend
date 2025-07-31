
import cv2
import pickle
import numpy as np

IMG_SIZE = 128
CATEGORIES = ["Healthy", "Yellowing", "Spotted", "Dried"]

# Load model
with open("model/leaf_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_leaf_condition(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().reshape(1, -1)
    prediction = model.predict(flat)[0]
    return f"The leaf is classified as: {CATEGORIES[prediction]}"
