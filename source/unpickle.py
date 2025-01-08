import os
import pickle
import tensorflow as tf


def load_saved_dataset(dataset_name):
    """
    Load the saved dataset and details from the dataset pickle file.
    """
    # Path to the dataset pickle file
    dataset_dir = os.path.join("dataset_info", dataset_name)
    pkl_path = os.path.join(dataset_dir, "data.pkl")

    # Check if the file exists
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Dataset pickle not found at {pkl_path}")
    
    # Load the pickle file
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)
    
    # Extract required elements
    X_train = data.get("X_train", None)
    X_test = data.get("X_test", None)
    y_train = data.get("y_train", None)
    y_test = data.get("y_test", None)
    input_shape = data.get("input_shape", None)
    num_classes = data.get("num_classes", None)
    
    # Validate the extracted data
    if any(var is None for var in [X_train, X_test, y_train, y_test, input_shape, num_classes]):
        raise ValueError("Missing required keys in the dataset pickle file.")
    
    return X_train, X_test, y_train, y_test, input_shape, num_classes


def load_median_image(dataset_name, target_digit):
    """
    Load the median image for a specific class (digit) from the saved pickle file.
    """
    # Path to the median images pickle file
    median_pkl_path = os.path.join("dataset_info", dataset_name, "aggregated_images", "median", "pkl", "median_images.pkl")
    
    print("Median image loading from: ", median_pkl_path)
    # Check if the file exists
    if not os.path.exists(median_pkl_path):
        raise FileNotFoundError(f"Median images pickle not found at {median_pkl_path}")
    

    # As median images were saved as a dictionary of images with image class name as key
    if dataset_name == "mnistDigits" or dataset_name == "sklearnDigits":
        key = target_digit  # e.g., 0..9
    elif dataset_name == "mnistFashion":
        # Convert 0..9 -> "T-shirt_or_Top", "Trouser", etc.
        fashion_mnist_classnames = [
            'T-shirt_or_Top',
            'Trouser',
            'Pullover',
            'Dress',
            'Coat',
            'Sandal',
            'Shirt',
            'Sneaker',
            'Bag',
            'Ankle boot'
        ]
        key = fashion_mnist_classnames[target_digit]

    # Load the pickle file
    with open(median_pkl_path, "rb") as file:
        median_images = pickle.load(file)
    
    # Validate the target digit
    if str(key) not in median_images:
        raise ValueError(f"Target digit {key} not found in the median images file.")
    
    return median_images[str(key)]


def load_best_image(dataset_name, model_name, target_digit, data_type):
    """
    Unpickle and return the best image saved for the specified dataset, model, and target digit.

    Parameters:
    - dataset_name (str): The name of the dataset (e.g., "sklearnDigits").
    - model_name (str): The name of the model (e.g., "CNN", "MLP").
    - target_digit (int): The target class/digit for which to load the best image.
    - data_type (str): "train" or "test" to specify whether to load from training or testing data.

    Returns:
    - dict: A dictionary containing the best image, confidence score, and class.
    """
    # Define the path to the best image pickle file
    image_dir = os.path.join("model_info", dataset_name, model_name, "best_images", data_type, "pkl")
    pkl_path = os.path.join(image_dir, f"class_{target_digit}_best_image.pkl")

    # Check if the file exists
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Best image pickle not found for class {target_digit} at {pkl_path}")

    # Load the best image data
    with open(pkl_path, "rb") as file:
        best_image_data = pickle.load(file)

    # Verify the structure of the loaded data
    if not all(key in best_image_data for key in ["image", "confidence", "class"]):
        raise ValueError(f"Best image pickle for class {target_digit} is missing required keys.")

    # Return the best image data
    return best_image_data

def load_trained_model(dataset_name, model_name):
    """
    Load the trained model for a specific dataset and model type.
    """
    # Path to the model directory
    model_dir = os.path.join("model_info", dataset_name, model_name)
    keras_model_path = os.path.join(model_dir, "trained_model.keras")
    pickle_model_path = os.path.join(model_dir, "trained_model.pkl")
    
    # Check if the model exists and load it
    if os.path.exists(keras_model_path):
        print(f"Loading Keras model from {keras_model_path}")
        return tf.keras.models.load_model(keras_model_path)
    elif os.path.exists(pickle_model_path):
        print(f"Loading pickled model from {pickle_model_path}")
        with open(pickle_model_path, "rb") as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"No trained model found for {model_name} in {model_dir}")

