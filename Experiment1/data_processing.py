import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

from skimage.io import imread


def load_dataset(dataset_name):
    """
    Load the specified dataset and return training and testing splits.
    """
    if dataset_name == "sklearnDigits":
        digits = datasets.load_digits()
        X = digits.images / 16.0
        y = np.eye(10)[digits.target]
        input_shape = (8, 8)
        num_classes = 10
    elif dataset_name == "mnistDigits":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test]) / 255.0
        y = np.concatenate([y_train, y_test])
        y = tf.keras.utils.to_categorical(y, num_classes=10)
        input_shape = (28, 28)
        num_classes = 10
    elif dataset_name == "mnistFashion":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        X = np.concatenate([X_train, X_test]) / 255.0
        y = np.concatenate([y_train, y_test])
        y = tf.keras.utils.to_categorical(y, num_classes=10)
        input_shape = (28, 28)
        num_classes = 10
    elif dataset_name == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X = np.concatenate([X_train, X_test]) / 255.0
        y = tf.keras.utils.to_categorical(np.concatenate([y_train, y_test]), num_classes=10)
        input_shape = (32, 32, 3)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        X = np.concatenate([X_train, X_test]) / 255.0
        y = tf.keras.utils.to_categorical(np.concatenate([y_train, y_test]), num_classes=100)
        input_shape = (32, 32, 3)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    return X, y, input_shape, num_classes

# Save mean and median images for reference
def save_mean_median_images(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for digit in range(10):
        digit_images = X[np.argmax(y, axis=1) == digit]
        mean_image = np.mean(digit_images, axis=0)
        median_image = np.median(digit_images, axis=0)
        plt.imsave(os.path.join(output_dir, f'digit_{digit}_mean.png'), mean_image, cmap='gray')
        plt.imsave(os.path.join(output_dir, f'digit_{digit}_median.png'), median_image, cmap='gray')

# Load the median image for a given digit
def load_median_image(output_dir, digit, num_channels=1):
    median_image_path = os.path.join(output_dir, f"digit_{digit}_median.png")
    median_image = imread(median_image_path)
    if num_channels == 1 and len(median_image.shape) == 3:  # Convert RGB to grayscale if needed
        median_image = np.mean(median_image, axis=-1)
    return median_image
