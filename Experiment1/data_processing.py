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

def evaluate_model_accuracy(model, model_name, X_train, y_train, X_test, y_test):
    """
    Evaluate overall and per-class accuracy of the model.
    """
    print("\nModel Evaluation:")
    
    # Predict and calculate overall accuracy
    if model_name in ["CNN", "RNN"]:
        train_pred = model.predict(X_train[..., np.newaxis])  # 2D predictions
        test_pred = model.predict(X_test[..., np.newaxis])    # 2D predictions
        train_pred = np.argmax(train_pred, axis=1)  # Convert to class indices
        test_pred = np.argmax(test_pred, axis=1)    # Convert to class indices
    else:
        train_pred = model.predict(X_train.reshape(X_train.shape[0], -1))  # 1D predictions for scikit-learn models
        test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    # Convert y_train and y_test to class indices
    y_train_indices = np.argmax(y_train, axis=1) if y_train.ndim == 2 else y_train
    y_test_indices = np.argmax(y_test, axis=1) if y_test.ndim == 2 else y_test
    
    # Overall accuracy
    train_acc = np.mean(train_pred == y_train_indices)
    test_acc = np.mean(test_pred == y_test_indices)
    
    print(f"Overall Train Accuracy: {train_acc:.4f}")
    print(f"Overall Test Accuracy: {test_acc:.4f}")
    
    # Per-class accuracy
    num_classes = y_train.shape[1] if y_train.ndim == 2 else len(np.unique(y_train))
    print("\nPer-Class Accuracy:")
    for cls in range(num_classes):
        train_cls_indices = y_train_indices == cls
        test_cls_indices = y_test_indices == cls
        train_cls_acc = np.mean(train_pred[train_cls_indices] == cls) if np.any(train_cls_indices) else 0
        test_cls_acc = np.mean(test_pred[test_cls_indices] == cls) if np.any(test_cls_indices) else 0
        print(f"Class {cls}: Train Accuracy = {train_cls_acc:.4f}, Test Accuracy = {test_cls_acc:.4f}")


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
    return median_image
