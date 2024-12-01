import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from skimage.io import imread

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
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)
        input_shape = (32, 32, 3)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    print_dataset_details(dataset_name, X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test, input_shape, num_classes

def print_dataset_details(dataset_name, X_train, X_test, y_train, y_test):
    """
    Print details of the dataset, including dimensions and number of samples.
    """
    print(f"\nDataset: {dataset_name}")
    print(f"Train Samples: {X_train.shape[0]}, Test Samples: {X_test.shape[0]}")
    print(f"Input Shape: {X_train.shape[1:]}")
    print(f"Number of Classes: {y_train.shape[1]}")


def save_mean_median_images(X, y, output_dir):
    """
    Save mean and median images for reference.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = y.shape[1]  # Number of classes from one-hot encoding
    for class_idx in range(num_classes):
        class_images = X[np.argmax(y, axis=1) == class_idx]
        if len(X.shape) == 4 and X.shape[-1] == 3:  # RGB Images
            mean_image = np.mean(class_images, axis=0)
            median_image = np.median(class_images, axis=0)
        else:  # Grayscale Images
            mean_image = np.mean(class_images, axis=0)
            median_image = np.median(class_images, axis=0)
        plt.imsave(os.path.join(output_dir, f'class_{class_idx}_mean.png'), mean_image, cmap='gray' if len(mean_image.shape) == 2 else None)
        plt.imsave(os.path.join(output_dir, f'class_{class_idx}_median.png'), median_image, cmap='gray' if len(median_image.shape) == 2 else None)

def load_median_image(output_dir, digit, num_channels=1):
    """
    Load the median image for a given class.
    """
    median_image_path = os.path.join(output_dir, f"class_{digit}_median.png")
    median_image = imread(median_image_path)
    if num_channels == 1 and len(median_image.shape) == 3:  # Convert RGB to grayscale if needed
        median_image = np.mean(median_image, axis=-1)
    elif num_channels == 3 and len(median_image.shape) == 2:  # Convert grayscale to RGB if needed
        median_image = np.stack([median_image] * 3, axis=-1)
    return median_image / np.max(median_image)
