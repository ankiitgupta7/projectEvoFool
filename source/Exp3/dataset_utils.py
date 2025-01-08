import os
import json
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from skimage.io import imsave

def fetch_class_labels(dataset_name):
    """
    Dynamically fetch class labels from the dataset source or provide defaults.
    """
    if dataset_name == "sklearnDigits":
        return {str(i): str(i) for i in range(10)}  # sklearn digits are numeric labels
    elif dataset_name == "mnistDigits":
        return {str(i): str(i) for i in range(10)}  # MNIST digits are numeric labels
    elif dataset_name == "mnistFashion":
        return {str(i): label for i, label in enumerate([
            "T-shirt_or_Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
            "Shirt", "Sneaker", "Bag", "Ankle boot"
        ])}
    elif dataset_name == "CIFAR10":
        return {str(i): label for i, label in enumerate([
            "Airplane", "Automobile", "Bird", "Cat", "Deer",
            "Dog", "Frog", "Horse", "Ship", "Truck"
        ])}
    elif dataset_name == "CIFAR100":
        fine_labels = [
            "Apple", "Aquarium fish", "Baby", "Bear", "Beaver", "Bed", "Bee", "Beetle", "Bicycle", "Bottle",
            "Bowl", "Boy", "Bridge", "Bus", "Butterfly", "Camel", "Can", "Castle", "Caterpillar", "Cattle",
            "Chair", "Chimpanzee", "Clock", "Cloud", "Cockroach", "Couch", "Crab", "Crocodile", "Cup", "Dinosaur",
            "Dolphin", "Elephant", "Flatfish", "Forest", "Fox", "Girl", "Hamster", "House", "Kangaroo", "Keyboard",
            "Lamp", "Lawn mower", "Leopard", "Lion", "Lizard", "Lobster", "Man", "Maple tree", "Motorcycle", "Mountain",
            "Mouse", "Mushroom", "Oak tree", "Orange", "Orchid", "Otter", "Palm tree", "Pear", "Pickup truck", "Pine tree",
            "Plain", "Plate", "Poppy", "Porcupine", "Possum", "Rabbit", "Raccoon", "Ray", "Road", "Rocket", "Rose",
            "Sea", "Seal", "Shark", "Shrew", "Skunk", "Skyscraper", "Snail", "Snake", "Spider", "Squirrel", "Streetcar",
            "Sunflower", "Sweet pepper", "Table", "Tank", "Telephone", "Television", "Tiger", "Tractor", "Train", "Trout",
            "Tulip", "Turtle", "Wardrobe", "Whale", "Willow tree", "Wolf", "Woman", "Worm"
        ]
        return {str(i): label for i, label in enumerate(fine_labels)}
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

def load_dataset(dataset_name, test_size=0.2, random_state=42):
    """
    Load the specified dataset, perform train-test split, and return details.
    """
    if dataset_name == "sklearnDigits":
        digits = datasets.load_digits()
        X = digits.images / 16.0
        y = np.eye(10)[digits.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        input_shape = (8, 8)
        num_classes = 10
    elif dataset_name == "mnistDigits":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
        input_shape = (28, 28)
        num_classes = 10
    elif dataset_name == "mnistFashion":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
        input_shape = (28, 28)
        num_classes = 10
    elif dataset_name == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
        input_shape = (32, 32, 3)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
        input_shape = (32, 32, 3)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    class_labels = fetch_class_labels(dataset_name)
    save_dataset_details(dataset_name, X_train, X_test, y_train, y_test, input_shape, num_classes, class_labels)
    return X_train, X_test, y_train, y_test, input_shape, num_classes

def save_images(X, y, output_dir, class_labels):
    """
    Save all images in PNG format, organized by class.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, (image, label_idx) in enumerate(zip(X, np.argmax(y, axis=1))):
        label_name = class_labels[str(label_idx)]
        filename = f"class_{label_idx}_{label_name}_img_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to uint8 format (0-255 range)
        image_uint8 = (image * 255).astype(np.uint8)
        
        cmap = "gray" if len(image.shape) == 2 else None
        imsave(filepath, image_uint8, cmap=cmap)

def save_mean_median_images(X, y, output_dir, class_labels):
    """
    Save mean and median images as PNG and Pickle in an organized structure.
    """
    mean_dir = os.path.join(output_dir, "mean")
    median_dir = os.path.join(output_dir, "median")
    os.makedirs(mean_dir, exist_ok=True)
    os.makedirs(median_dir, exist_ok=True)
    
    mean_png_dir = os.path.join(mean_dir, "png")
    mean_pkl_dir = os.path.join(mean_dir, "pkl")
    median_png_dir = os.path.join(median_dir, "png")
    median_pkl_dir = os.path.join(median_dir, "pkl")
    
    os.makedirs(mean_png_dir, exist_ok=True)
    os.makedirs(mean_pkl_dir, exist_ok=True)
    os.makedirs(median_png_dir, exist_ok=True)
    os.makedirs(median_pkl_dir, exist_ok=True)
    
    mean_dict = {}
    median_dict = {}
    
    for class_idx in range(y.shape[1]):
        class_images = X[np.argmax(y, axis=1) == class_idx]
        label_name = class_labels[str(class_idx)]
        
        mean_image = np.mean(class_images, axis=0)
        median_image = np.median(class_images, axis=0)
        
        # Convert to uint8 format
        mean_image_uint8 = (mean_image * 255).astype(np.uint8)
        median_image_uint8 = (median_image * 255).astype(np.uint8)
        
        mean_png_path = os.path.join(mean_png_dir, f"class_{class_idx}_{label_name}_mean.png")
        median_png_path = os.path.join(median_png_dir, f"class_{class_idx}_{label_name}_median.png")
        plt.imsave(mean_png_path, mean_image_uint8, cmap="gray" if mean_image.ndim == 2 else None)
        plt.imsave(median_png_path, median_image_uint8, cmap="gray" if median_image.ndim == 2 else None)
        
        mean_dict[label_name] = mean_image
        median_dict[label_name] = median_image
    
    with open(os.path.join(mean_pkl_dir, "mean_images.pkl"), "wb") as mean_pkl_file:
        pickle.dump(mean_dict, mean_pkl_file)
    
    with open(os.path.join(median_pkl_dir, "median_images.pkl"), "wb") as median_pkl_file:
        pickle.dump(median_dict, median_pkl_file)


def save_dataset_details(dataset_name, X_train, X_test, y_train, y_test, input_shape, num_classes, class_labels):
    output_dir = os.path.join("dataset_info", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "Dataset Name": dataset_name,
        "Train Samples": X_train.shape[0],
        "Test Samples": X_test.shape[0],
        "Dataset Dimension": input_shape,
        "Number of Classes": num_classes,
        "Pixel Range": f"{X_train.min()} to {X_train.max()}",
        "Image Type": "Colored" if len(input_shape) == 3 else "Grayscale",
        "Labels": class_labels
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as json_file:
        json.dump(metadata, json_file, indent=4)

    # Save train and test images
    save_images(X_train, y_train, os.path.join(output_dir, "train_images"), class_labels)
    save_images(X_test, y_test, os.path.join(output_dir, "test_images"), class_labels)

    # Save mean and median images
    mean_median_dir = os.path.join(output_dir, "aggregated_images")
    save_mean_median_images(X_train, y_train, mean_median_dir, class_labels)

    # Save all dataset details to data.pkl
    with open(os.path.join(output_dir, "data.pkl"), "wb") as pkl_file:
        pickle.dump({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "input_shape": input_shape,
            "num_classes": num_classes
        }, pkl_file)

datasets_to_load = ["sklearnDigits", "mnistDigits", "mnistFashion", "CIFAR10", "CIFAR100"]
loaded_datasets = {dataset: load_dataset(dataset) for dataset in datasets_to_load}
