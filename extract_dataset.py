import tarfile
import os

def extract_dataset(tar_path, extract_path):
    print(f"Extracting {tar_path} to {extract_path}...")
    if not os.path.exists(tar_path):
        print(f"Error: {tar_path} not found.")
        return

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    except Exception as e:
        print(f"An error occurred during extraction: {e}")

if __name__ == "__main__":
    tar_path = os.path.join("dataset", "food-101.tar.gz")
    extract_path = "dataset"
    extract_dataset(tar_path, extract_path)
