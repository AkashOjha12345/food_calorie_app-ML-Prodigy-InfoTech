import os
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE

def load_data(train_dir, test_dir, max_classes=None):
    all_class_names = sorted(os.listdir(train_dir))
    if max_classes is not None:
        selected_classes = all_class_names[:max_classes]
    else:
        selected_classes = all_class_names

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        class_names=selected_classes,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        class_names=selected_classes,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_data.class_names

    return train_data, test_data, class_names