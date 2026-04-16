import json
import os

import cv2
import numpy as np
import tensorflow as tf

from config import CLASS_NAMES_PATH, IMG_SIZE, MODEL_PATH

model = None
model_load_error = None
class_names_load_error = None

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)

class_names = []


def load_saved_class_names():
    global class_names
    global class_names_load_error

    if not os.path.exists(CLASS_NAMES_PATH):
        class_names = []
        class_names_load_error = (
            f"Saved class labels were not found at '{CLASS_NAMES_PATH}'."
        )
        return

    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = json.load(file)
        class_names_load_error = None
    except Exception as exc:
        class_names = []
        class_names_load_error = str(exc)


load_saved_class_names()

def load_class_names(names):
    global class_names
    global class_names_load_error
    class_names = names
    class_names_load_error = None

def is_model_ready():
    return model is not None

def get_model_error():
    return model_load_error


def are_class_names_ready():
    return bool(class_names)


def get_class_names_error():
    return class_names_load_error

def predict_image(img_path):
    if model is None:
        raise RuntimeError(
            f"Model is not available. Expected trained model at '{MODEL_PATH}'."
        )

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at '{img_path}'.")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(
            f"Failed to load image at '{img_path}'. The file may be corrupt or not a valid image."
        )

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    if not class_names:
        raise RuntimeError(
            "Class labels are not available. Train the model again to generate them."
        )

    return class_names[class_id], confidence
