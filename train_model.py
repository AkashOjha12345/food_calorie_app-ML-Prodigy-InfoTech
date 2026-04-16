import json

import tensorflow as tf

from config import CLASS_NAMES_PATH, EPOCHS, MODEL_PATH, TEST_DIR, TRAIN_DIR
from src.data_loader import load_data

# For demonstration, limit to 5 classes and 1 epoch
MAX_CLASSES = 5
EPOCHS = 1

train, test, class_names = load_data(TRAIN_DIR, TEST_DIR, max_classes=MAX_CLASSES)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train, validation_data=test, epochs=EPOCHS)

model.save(MODEL_PATH)
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file)

print("Model trained and saved.")
