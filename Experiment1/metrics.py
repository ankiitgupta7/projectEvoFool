import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template

# SSIM computation function
def compute_ssim(image1, image2):
    return ssim(image1, image2, data_range=1.0)

# NCC computation function
def compute_ncc(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same dimensions")
    result = match_template(image1, image2)
    return result.max()
