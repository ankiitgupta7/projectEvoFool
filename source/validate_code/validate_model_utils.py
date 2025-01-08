import os
import pickle
import json
import csv


def validate_pickle(pkl_path, required_keys=None):
    """
    Validate the structure and presence of a pickle file.
    """
    if not os.path.exists(pkl_path):
        return f"ERROR: Pickle file not found at {pkl_path}"
    try:
        with open(pkl_path, "rb") as file:
            data = pickle.load(file)
        if required_keys and not all(key in data for key in required_keys):
            return f"ERROR: Missing required keys in {pkl_path}"
        return f"SUCCESS: Pickle file {pkl_path} loaded successfully"
    except Exception as e:
        return f"ERROR: Failed to load {pkl_path} - {str(e)}"


def validate_json(json_path):
    """
    Validate the presence and correctness of a JSON file.
    """
    if not os.path.exists(json_path):
        return f"ERROR: JSON file not found: {json_path}"
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
        return f"SUCCESS: JSON file {json_path} loaded successfully"
    except Exception as e:
        return f"ERROR: Failed to load {json_path} - {str(e)}"


def validate_csv(csv_path):
    """
    Validate the presence and structure of a CSV file.
    """
    if not os.path.exists(csv_path):
        return f"ERROR: CSV file not found: {csv_path}"
    try:
        with open(csv_path, "r") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if not header:
                return f"ERROR: CSV file is empty: {csv_path}"
        return f"SUCCESS: CSV file {csv_path} loaded successfully"
    except Exception as e:
        return f"ERROR: Failed to load {csv_path} - {str(e)}"


def validate_best_images(image_dir, data_type, num_classes):
    """
    Validate the presence of best images for each class in both .pkl and .png formats,
    considering the subdirectories for `train` and `test`.
    """
    errors = []
    pkl_dir = os.path.join(image_dir, data_type, "pkl")
    png_dir = os.path.join(image_dir, data_type, "png")

    for cls in range(num_classes):
        pkl_path = os.path.join(pkl_dir, f"class_{cls}_best_image.pkl")
        png_path = os.path.join(png_dir, f"class_{cls}_best_image.png")

        if not os.path.exists(pkl_path):
            errors.append(f"ERROR: Missing best image pickle for class {cls}: {pkl_path}")
        if not os.path.exists(png_path):
            errors.append(f"ERROR: Missing best image PNG for class {cls}: {png_path}")
    return errors if errors else [f"SUCCESS: All best images in {data_type} validated successfully"]


def validate_model_outputs(dataset_name, model_name):
    """
    Validate all outputs for a specific model and dataset.
    """
    output_dir = os.path.join("model_info", dataset_name, model_name)
    summary = []

    # Validate model parameters
    model_params_path = os.path.join(output_dir, "model_params.json")
    summary.append(validate_json(model_params_path))

    # Validate confidence scores
    train_conf_path = os.path.join(output_dir, "train_confidences.csv")
    test_conf_path = os.path.join(output_dir, "test_confidences.csv")
    summary.append(validate_csv(train_conf_path))
    summary.append(validate_csv(test_conf_path))

    # Validate accuracy files
    overall_acc_path = os.path.join(output_dir, "overall_accuracy.json")
    per_class_acc_path = os.path.join(output_dir, "per_class_accuracy.json")
    summary.append(validate_json(overall_acc_path))
    summary.append(validate_json(per_class_acc_path))

    # Infer the number of classes
    try:
        with open(overall_acc_path, "r") as file:
            overall_data = json.load(file)
        num_classes = overall_data.get("num_classes", None)
    except Exception:
        num_classes = None

    if num_classes is None:
        try:
            with open(per_class_acc_path, "r") as file:
                per_class_data = json.load(file)
            num_classes = len(per_class_data)
        except Exception:
            summary.append("ERROR: Unable to infer number of classes from accuracy files")
            num_classes = 0

    # Validate best images
    if num_classes > 0:
        best_images_dir = os.path.join(output_dir, "best_images")
        summary.extend(validate_best_images(best_images_dir, "train", num_classes))
        summary.extend(validate_best_images(best_images_dir, "test", num_classes))

    return summary


def validate_all_models(dataset_name, model_names):
    """
    Validate outputs for all models for a specific dataset.
    """
    print(f"Validating dataset: {dataset_name}")
    dataset_dir = os.path.join("dataset_info", dataset_name)
    data_pkl_path = os.path.join(dataset_dir, "data.pkl")

    # Validate dataset pickle
    summary = [validate_pickle(data_pkl_path, required_keys=["X_train", "X_test", "y_train", "y_test", "input_shape", "num_classes"])]

    for model_name in model_names:
        print(f"\nValidating model: {model_name}")
        model_summary = validate_model_outputs(dataset_name, model_name)
        summary.extend(model_summary)

    # Print summary
    print("\nValidation Summary:")
    for message in summary:
        print(message)


if __name__ == "__main__":
    # Specify dataset and models
    dataset_name = "sklearnDigits"  # Replace with your dataset name
    model_names = ["CNN", "RNN", "SVM", "RF", "GBM", "MLP"]  # Replace with models to validate

    validate_all_models(dataset_name, model_names)
