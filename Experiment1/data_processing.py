import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

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
def load_median_image(output_dir, digit):
    median_image_path = os.path.join(output_dir, f'digit_{digit}_median.png')
    return imread(median_image_path, as_gray=True)
