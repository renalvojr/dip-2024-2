# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np


def translate(img: np.ndarray, tx: int = 10, ty: int = 10) -> np.ndarray:
    """
    Shift the image right by tx and down by ty, filling empty regions with zeros.
    """
    h, w = img.shape
    result = np.zeros_like(img)

    ys, xs = np.indices((h, w))
    src_y = ys - ty
    src_x = xs - tx
    mask = (src_y >= 0) & (src_y < h) & (src_x >= 0) & (src_x < w)
    result[ys[mask], xs[mask]] = img[src_y[mask], src_x[mask]]
    return result


def rotate90(img: np.ndarray) -> np.ndarray:
    """
    Rotate the image 90 degrees clockwise.
    """
    return np.rot90(img, k=-1)


def stretch_horizontal(img: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """
    Stretch the image horizontally by a given scale (nearest-neighbor).
    """
    h, w = img.shape
    new_w = int(np.round(w * scale))
    result = np.zeros((h, new_w), dtype=img.dtype)
    xs = np.arange(new_w)
    src_x = np.clip(np.round(xs / scale).astype(int), 0, w - 1)

    for y in range(h):
        result[y, :] = img[y, src_x]
    return result


def mirror_horizontal(img: np.ndarray) -> np.ndarray:
    """
    Horizontally mirror (flip) the image along the vertical axis.
    """
    return img[:, ::-1]


def barrel_distort(img: np.ndarray, k: float = -0.3) -> np.ndarray:
    """
    Apply a simple barrel distortion using a radial mapping:
        r_distorted = r * (1 + k * r^2)
    where r is the normalized radius from center.
    """
    h, w = img.shape
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.indices((h, w))

    x_norm = (xs - cx) / cx
    y_norm = (ys - cy) / cy
    r = np.sqrt(x_norm**2 + y_norm**2)

    r_dist = r * (1 + k * r**2)

    x_dist = x_norm * r_dist * cx + cx
    y_dist = y_norm * r_dist * cy + cy

    x_src = np.clip(np.round(x_dist).astype(int), 0, w - 1)
    y_src = np.clip(np.round(y_dist).astype(int), 0, h - 1)

    return img[y_src, x_src]


def apply_geometric_transformations(img: np.ndarray) -> dict:
    """
    Apply a suite of geometric transforms to a grayscale image.
    Returns a dict with keys: translated, rotated, stretched, mirrored, distorted.
    """
    if img.ndim != 2:
        raise ValueError("Input must be a 2D grayscale image")

    return {
        "translated": translate(img),
        "rotated": rotate90(img),
        "stretched": stretch_horizontal(img),
        "mirrored": mirror_horizontal(img),
        "distorted": barrel_distort(img)
    }