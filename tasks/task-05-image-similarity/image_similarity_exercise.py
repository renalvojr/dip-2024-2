# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np


def mse(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Compute the Mean Squared Error between two images.
    """
    return np.mean((i1 - i2) ** 2)


def psnr(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio between two images.
    Assumes images are normalized in [0, 1].
    """
    mse_val = mse(i1, i2)
    if mse_val == 0:
        return float('inf')
    max_i = 1.0
    return 10 * np.log10((max_i ** 2) / mse_val)


def ssim(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Compute a simplified Structural Similarity Index between two images.
    """
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))

    K1, K2 = 0.01, 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


def npcc(i1: np.ndarray, i2: np.ndarray) -> float:
    """
    Compute the Normalized Pearson Correlation Coefficient between two images.
    """
    f1 = i1.flatten()
    f2 = i2.flatten()
    mu1, mu2 = np.mean(f1), np.mean(f2)

    num = np.sum((f1 - mu1) * (f2 - mu2))
    den = np.sqrt(np.sum((f1 - mu1) ** 2) * np.sum((f2 - mu2) ** 2))
    if den == 0:
        return 1.0 if np.allclose(i1, i2) else 0.0
    return num / den


def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    """
    Compare two normalized grayscale images by computing similarity metrics.
    Returns a dictionary with keys: mse, psnr, ssim, npcc.
    """
    if i1.shape != i2.shape:
        raise ValueError("Input images must have the same dimensions")

    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }
