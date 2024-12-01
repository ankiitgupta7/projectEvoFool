import os
import json
import pickle
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.io import imsave
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# Load dataset
def load_saved_dataset(dataset_name):
    """
    Load the saved dataset and details from dataset_utils outputs.
    """
    dataset_dir = os.path.join("dataset_info", dataset_name)
    pkl_path = os.path.join(dataset_dir, "data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Dataset pickle not found at {pkl_path}")
    
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)
    
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"], data["input_shape"], data["num_classes"]

# Model initialization
def get_model(model_name, input_shape):
    """
    Returns the appropriate model based on the model_name.
    """
    if model_name == "CNN":
        return create_cnn_model(input_shape)
    elif model_name == "RNN":
        return create_rnn_model(input_shape)
    elif model_name == "SVM":
        return SVC(gamma="scale", probability=True)
    elif model_name == "RF":
        return RandomForestClassifier()
    elif model_name == "GBM":
        return GradientBoostingClassifier()
    elif model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# CNN model creation
def create_cnn_model(input_shape):
    """
    Creates and compiles a CNN model.
    """
    if len(input_shape) == 2:  # Grayscale
        input_shape = (*input_shape, 1)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# RNN model creation
def create_rnn_model(input_shape):
    """
    Creates and compiles an RNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(50, return_sequences=True),
        tf.keras.layers.SimpleRNN(50),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train and evaluate model
def train_and_evaluate_model(dataset_name, model_name):
    """
    Train the model and evaluate its performance.
    """
    print(f"Loading dataset: {dataset_name}")
    X_train, X_test, y_train, y_test, input_shape, num_classes = load_saved_dataset(dataset_name)
    
    print(f"Dataset loaded successfully. Shape of training data: {X_train.shape}, testing data: {X_test.shape}")
    print(f"Input shape: {input_shape}, Number of classes: {num_classes}")
    
    # Initialize the model
    print(f"Initializing model: {model_name}")
    model = get_model(model_name, input_shape)
    
    # Train the model
    print("Training the model...")
    if model_name in ["CNN", "RNN"]:
        if len(X_train.shape) == 3:
            X_train = X_train[..., np.newaxis]
            X_test = X_test[..., np.newaxis]
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    else:
        model.fit(X_train.reshape((X_train.shape[0], -1)), y_train.argmax(axis=1))
    print("Training completed.")
    
    # Evaluate and save model information
    print("Evaluating model and saving details...")
    save_model_info(dataset_name, model, model_name, input_shape, X_train, y_train, X_test, y_test)
    print(f"Model information saved under model_info/{dataset_name}/{model_name}")

# Save model information
def save_model_info(dataset_name, model, model_name, input_shape, X_train, y_train, X_test, y_test):
    """
    Save model-related information, organized by dataset and model.
    """
    output_dir = os.path.join("model_info", dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save model parameters
    if model_name in ["CNN", "RNN"]:
        model_params = {
            "model_type": model_name,
            "architecture": model.to_json(),
            "input_shape": input_shape
        }
    else:
        model_params = {
            "model_type": model_name,
            "parameters": model.get_params()
        }
    with open(os.path.join(output_dir, "model_params.json"), "w") as json_file:
        json.dump(model_params, json_file, indent=4)

    # Save the trained model
    save_trained_model(model, model_name, output_dir)

    # Evaluate and save confidence scores
    train_probs, test_probs = evaluate_and_save_confidences(
        model, model_name, X_train, y_train, X_test, y_test, output_dir
    )
    
    # Calculate overall and per-class accuracy
    save_accuracy_details(train_probs, test_probs, y_train, y_test, output_dir)
    
    # Save best images for each class
    save_best_images_per_class(X_train, train_probs, y_train, "train", model_name, output_dir)
    save_best_images_per_class(X_test, test_probs, y_test, "test", model_name, output_dir)

# Save trained model
def save_trained_model(model, model_name, output_dir):
    """
    Save the trained model for reuse.
    """
    if model_name in ["CNN", "RNN"]:
        model.save(os.path.join(output_dir, "trained_model.keras"))
    else:
        with open(os.path.join(output_dir, "trained_model.pkl"), "wb") as file:
            pickle.dump(model, file)

# Evaluate and save confidences
def evaluate_and_save_confidences(model, model_name, X_train, y_train, X_test, y_test, output_dir):
    """
    Evaluate model and save confidence scores for train/test datasets.
    """
    if model_name in ["CNN", "RNN"]:
        train_probs = model.predict(X_train)
        test_probs = model.predict(X_test)
    else:
        train_probs = model.predict_proba(X_train.reshape((X_train.shape[0], -1)))
        test_probs = model.predict_proba(X_test.reshape((X_test.shape[0], -1)))
    
    save_confidences_to_csv(train_probs, y_train, os.path.join(output_dir, "train_confidences.csv"))
    save_confidences_to_csv(test_probs, y_test, os.path.join(output_dir, "test_confidences.csv"))
    
    return train_probs, test_probs

# Save confidence scores to CSV
def save_confidences_to_csv(probs, y_data, file_path):
    """
    Save confidence scores for all classes in a CSV file.
    """
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index", "True Class"] + [f"Confidence for Class {i}" for i in range(probs.shape[1])])
        for idx, (true_class, prob) in enumerate(zip(np.argmax(y_data, axis=1), probs)):
            writer.writerow([idx, true_class] + list(prob))

# Save accuracy details
def save_accuracy_details(train_probs, test_probs, y_train, y_test, output_dir):
    """
    Save overall and per-class accuracy details.
    """
    train_preds = np.argmax(train_probs, axis=1)
    test_preds = np.argmax(test_probs, axis=1)
    y_train_indices = np.argmax(y_train, axis=1)
    y_test_indices = np.argmax(y_test, axis=1)
    
    overall_accuracy = {
        "train_accuracy": np.mean(train_preds == y_train_indices),
        "test_accuracy": np.mean(test_preds == y_test_indices),
    }
    with open(os.path.join(output_dir, "overall_accuracy.json"), "w") as json_file:
        json.dump(overall_accuracy, json_file, indent=4)
    
    # Per-class accuracy
    per_class_accuracy = {}
    for cls in range(y_train.shape[1]):
        train_cls_indices = y_train_indices == cls
        test_cls_indices = y_test_indices == cls
        train_cls_acc = np.mean(train_preds[train_cls_indices] == cls) if np.any(train_cls_indices) else 0
        test_cls_acc = np.mean(test_preds[test_cls_indices] == cls) if np.any(test_cls_indices) else 0
        per_class_accuracy[str(cls)] = {"train_accuracy": train_cls_acc, "test_accuracy": test_cls_acc}
    with open(os.path.join(output_dir, "per_class_accuracy.json"), "w") as json_file:
        json.dump(per_class_accuracy, json_file, indent=4)

# Save best images for all classes
def save_best_images_per_class(X_data, probs, y_data, data_type, model_name, output_dir):
    """
    Save the best images (highest confidence) for each class in both test and train sets.
    """
    y_indices = np.argmax(y_data, axis=1)
    num_classes = y_data.shape[1]
    best_images_dir = os.path.join(output_dir, "best_images", data_type)
    pkl_dir = os.path.join(best_images_dir, "pkl")
    png_dir = os.path.join(best_images_dir, "png")
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for cls in range(num_classes):
        cls_indices = np.where(y_indices == cls)[0]
        if cls_indices.size == 0:
            continue
        cls_probs = probs[cls_indices, cls]
        best_idx = cls_indices[np.argmax(cls_probs)]
        best_image = X_data[best_idx]
        confidence = cls_probs.max()

        # Save the image and metadata as a pickle
        best_image_data = {
            "image": best_image,
            "confidence": confidence,
            "class": cls,
        }
        with open(os.path.join(pkl_dir, f"class_{cls}_best_image.pkl"), "wb") as pkl_file:
            pickle.dump(best_image_data, pkl_file)
        
        # Render and save the image
        cmap = "gray" if best_image.ndim == 2 or (best_image.ndim == 3 and best_image.shape[-1] == 1) else None
        plt.imshow(best_image.squeeze(), cmap=cmap)
        plt.axis("off")
        plt.title(f"Class: {cls}\nConfidence: {confidence:.4f}\nModel: {model_name}")
        plt.savefig(os.path.join(png_dir, f"class_{cls}_best_image.png"))
        plt.close()


# Run the model processing pipeline
if __name__ == "__main__":
    dataset_name = "Digits"  # Replace with your dataset name
    model_names = ["CNN", "RNN", "SVM", "RF", "GBM", "MLP"]  # List of models to test

    for model_name in model_names:
        print(f"Processing model: {model_name}")
        train_and_evaluate_model(dataset_name, model_name)
