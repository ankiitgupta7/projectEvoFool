import os
import pickle
import json
import numpy as np


def test_pkl_integrity(pkl_path, expected_keys=None):
    """
    Test the integrity of a pickle file.
    """
    if not os.path.exists(pkl_path):
        return f"ERROR: Pickle file not found: {pkl_path}"
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if expected_keys and not all(key in data for key in expected_keys):
            return f"ERROR: Pickle file missing expected keys: {pkl_path}"
        return f"SUCCESS: {pkl_path} loaded correctly"
    except Exception as e:
        return f"ERROR: Failed to load pickle file {pkl_path} - {str(e)}"


def test_json_integrity(json_path):
    """
    Test the integrity of a JSON file.
    """
    if not os.path.exists(json_path):
        return f"ERROR: JSON file not found: {json_path}"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return f"ERROR: JSON file does not contain a dictionary: {json_path}"
        return f"SUCCESS: {json_path} loaded correctly"
    except Exception as e:
        return f"ERROR: Failed to load JSON file {json_path} - {str(e)}"


def test_images_integrity(image_dir, num_classes, class_labels):
    """
    Test if images are saved correctly for each class.
    """
    if not os.path.exists(image_dir):
        return f"ERROR: Image directory not found: {image_dir}"
    image_files = os.listdir(image_dir)
    images_per_class = {label: 0 for label in class_labels.values()}

    for image_name in image_files:
        try:
            class_idx = image_name.split("_")[1]
            class_name = class_labels.get(class_idx)
            if not class_name:
                return f"ERROR: Unknown class label in image name: {image_name}"
            images_per_class[class_name] += 1
        except Exception as e:
            return f"ERROR: Issue parsing image name {image_name} - {str(e)}"

    missing_classes = [label for label, count in images_per_class.items() if count == 0]
    if missing_classes:
        return f"ERROR: Missing images for classes: {missing_classes}"
    return f"SUCCESS: All {len(image_files)} images verified in {image_dir}"


def test_mean_median_integrity(mean_pkl_path, median_pkl_path, class_labels):
    """
    Test if mean and median pickle files are saved and match the class labels.
    """
    mean_check = test_pkl_integrity(mean_pkl_path)
    median_check = test_pkl_integrity(median_pkl_path)

    if "ERROR" in mean_check or "ERROR" in median_check:
        return mean_check, median_check

    with open(mean_pkl_path, "rb") as f:
        mean_data = pickle.load(f)
    with open(median_pkl_path, "rb") as f:
        median_data = pickle.load(f)

    mean_classes = set(mean_data.keys())
    median_classes = set(median_data.keys())
    expected_classes = set(class_labels.values())

    missing_in_mean = expected_classes - mean_classes
    missing_in_median = expected_classes - median_classes

    if missing_in_mean or missing_in_median:
        return (
            f"ERROR: Missing classes in mean pickle: {missing_in_mean}" if missing_in_mean else "SUCCESS: Mean pickle verified",
            f"ERROR: Missing classes in median pickle: {missing_in_median}" if missing_in_median else "SUCCESS: Median pickle verified",
        )
    return "SUCCESS: Mean pickle verified", "SUCCESS: Median pickle verified"


def test_dataset(dataset_name):
    """
    Test all aspects of a saved dataset.
    """
    print(f"--- Testing dataset: {dataset_name} ---")
    dataset_dir = os.path.join("dataset_info", dataset_name)

    # Test metadata.json
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    metadata_result = test_json_integrity(metadata_path)
    print(metadata_result)
    if "ERROR" in metadata_result:
        return

    # Load metadata for further checks
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Test data.pkl
    data_pkl_path = os.path.join(dataset_dir, "data.pkl")
    data_result = test_pkl_integrity(data_pkl_path, expected_keys=["X_train", "X_test", "y_train", "y_test", "input_shape", "num_classes"])
    print(data_result)

    # Test train and test images
    train_images_dir = os.path.join(dataset_dir, "train_images")
    test_images_dir = os.path.join(dataset_dir, "test_images")
    train_images_result = test_images_integrity(train_images_dir, metadata["Number of Classes"], metadata["Labels"])
    test_images_result = test_images_integrity(test_images_dir, metadata["Number of Classes"], metadata["Labels"])
    print(train_images_result)
    print(test_images_result)

    # Test mean and median pickle files
    mean_pkl_path = os.path.join(dataset_dir, "aggregated_images", "mean", "pkl", "mean_images.pkl")
    median_pkl_path = os.path.join(dataset_dir, "aggregated_images", "median", "pkl", "median_images.pkl")
    mean_result, median_result = test_mean_median_integrity(mean_pkl_path, median_pkl_path, metadata["Labels"])
    print(mean_result)
    print(median_result)

    print(f"--- Testing completed for {dataset_name} ---\n")


def test_all_datasets(datasets):
    """
    Test all datasets in the given list.
    """
    print("--- Starting Dataset Testing ---")
    for dataset_name in datasets:
        test_dataset(dataset_name)
    print("--- All Dataset Tests Completed ---")


if __name__ == "__main__":
    datasets_to_test = ["sklearnDigits", "mnistDigits", "mnistFashion", "CIFAR10", "CIFAR100"]
    test_all_datasets(datasets_to_test)
