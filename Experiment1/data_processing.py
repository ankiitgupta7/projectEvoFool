import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from skimage.io import imread

def load_data():
    digits = datasets.load_digits()
    X = digits.images / 16.0
    y = np.eye(10)[digits.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_shape = X_train[0].shape
    return X_train, X_test, y_train, y_test, input_shape

def save_mean_median_images(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for digit in range(10):
        digit_images = X[np.argmax(y, axis=1) == digit]
        mean_image = np.mean(digit_images, axis=0)
        median_image = np.median(digit_images, axis=0)
        plt.imsave(os.path.join(output_dir, f"digit_{digit}_mean.png"), mean_image, cmap="gray")
        plt.imsave(os.path.join(output_dir, f"digit_{digit}_median.png"), median_image, cmap="gray")

def load_median_image(output_dir, digit):
    median_image_path = os.path.join(output_dir, f"digit_{digit}_median.png")
    return imread(median_image_path, as_gray=True)
