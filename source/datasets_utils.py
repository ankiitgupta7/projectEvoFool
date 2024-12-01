import os
import json
import csv
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
        # Dynamically fetch labels for Fashion MNIST
        return {str(i): label for i, label in enumerate([
            "T-shirt_or_Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
            "Shirt", "Sneaker", "Bag", "Ankle boot"
        ])}
    elif dataset_name == "CIFAR10":
        # Dynamically fetch labels for CIFAR-10
        return {str(i): label for i, label in enumerate([
            "Airplane", "Automobile", "Bird", "Cat", "Deer",
            "Dog", "Frog", "Horse", "Ship", "Truck"
        ])}
    elif dataset_name == "CIFAR100":
        # Dynamically fetch labels for CIFAR-100
        cifar100_metadata = tf.keras.datasets.cifar100.load_data(label_mode="fine")
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

    # Fetch class labels
    class_labels = fetch_class_labels(dataset_name)

    # Save dataset details
    save_dataset_details(
        dataset_name, X_train, X_test, y_train, y_test, input_shape, num_classes, class_labels
    )
    return X_train, X_test, y_train, y_test, input_shape, num_classes

def save_image(image, class_idx, label_name, output_dir, idx):
    """
    Save a single image with the class index, label name, and index in the filename.
    """
    # Normalize the image to range [0, 255] and convert to uint8
    normalized_image = (image - image.min()) / (image.max() - image.min()) * 255
    normalized_image = normalized_image.astype(np.uint8)

    # Define file name and save the image
    filename = f"class_{class_idx}_{label_name}_img_{idx}.png"
    image_path = os.path.join(output_dir, filename)
    cmap = "gray" if len(image.shape) == 2 else None
    imsave(image_path, normalized_image, cmap=cmap)

def save_mean_median_images(X, y, output_dir, class_labels):
    """
    Save mean and median images for each class.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = y.shape[1]  # Number of classes from one-hot encoding
    for class_idx in range(num_classes):
        class_images = X[np.argmax(y, axis=1) == class_idx]
        label_name = class_labels[str(class_idx)]
        mean_image = np.mean(class_images, axis=0)
        median_image = np.median(class_images, axis=0)
        mean_filename = f"class_{class_idx}_{label_name}_mean.png"
        median_filename = f"class_{class_idx}_{label_name}_median.png"
        plt.imsave(os.path.join(output_dir, mean_filename), mean_image, cmap="gray" if len(mean_image.shape) == 2 else None)
        plt.imsave(os.path.join(output_dir, median_filename), median_image, cmap="gray" if len(median_image.shape) == 2 else None)

def save_dataset_details(dataset_name, X_train, X_test, y_train, y_test, input_shape, num_classes, class_labels):
    """
    Save dataset details and organize files for reuse.
    """
    output_dir = os.path.join("dataset_info", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata as JSON
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

    # Save train/test images in separate folders
    train_images_dir = os.path.join(output_dir, "train_images")
    test_images_dir = os.path.join(output_dir, "test_images")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    for idx, (image, label_idx) in enumerate(zip(X_train, np.argmax(y_train, axis=1))):
        label_name = class_labels[str(label_idx)]
        save_image(image, label_idx, label_name, train_images_dir, idx)

    for idx, (image, label_idx) in enumerate(zip(X_test, np.argmax(y_test, axis=1))):
        label_name = class_labels[str(label_idx)]
        save_image(image, label_idx, label_name, test_images_dir, idx)

    # Save mean and median images with label names
    mean_median_dir = os.path.join(output_dir, "mean_median")
    save_mean_median_images(X_train, y_train, mean_median_dir, class_labels)

    # Save data as pickle for faster reuse
    data_path = os.path.join(output_dir, "data.pkl")
    with open(data_path, "wb") as pkl_file:
        pickle.dump(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "input_shape": input_shape,
                "num_classes": num_classes,
            },
            pkl_file,
        )


datasets_to_load = ["sklearnDigits", "mnistDigits", "mnistFashion", "CIFAR10", "CIFAR100"]
loaded_datasets = {}

for dataset in datasets_to_load:
    loaded_datasets[dataset] = load_dataset(dataset)