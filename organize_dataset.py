import os
import shutil
import random

def organize_dataset(source_dir, train_dir, test_dir, split_ratio=0.8):
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found.")
        return

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(classes)} classes.")

    for cls in classes:
        cls_source_dir = os.path.join(source_dir, cls)
        cls_train_dir = os.path.join(train_dir, cls)
        cls_test_dir = os.path.join(test_dir, cls)

        os.makedirs(cls_train_dir, exist_ok=True)
        os.makedirs(cls_test_dir, exist_ok=True)

        images = [f for f in os.listdir(cls_source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for img in train_images:
            shutil.copy(os.path.join(cls_source_dir, img), os.path.join(cls_train_dir, img))
        
        for img in test_images:
            shutil.copy(os.path.join(cls_source_dir, img), os.path.join(cls_test_dir, img))

        print(f"Organized class {cls}: {len(train_images)} train, {len(test_images)} test.")

if __name__ == "__main__":
    source_dir = os.path.join("dataset", "food-101", "images")
    train_dir = os.path.join("dataset", "train")
    test_dir = os.path.join("dataset", "test")
    organize_dataset(source_dir, train_dir, test_dir)
    print("Dataset organization complete.")
