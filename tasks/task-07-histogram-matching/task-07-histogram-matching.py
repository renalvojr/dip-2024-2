# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched_img to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched_img image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import numpy as np

def match_histograms_rgb(source_img: np.ndarray,
                         reference_img: np.ndarray) -> np.ndarray:
    matched_img = np.zeros_like(source_img, dtype=np.uint8)

    for ch in range(3):
        src = source_img[:, :, ch].ravel()
        ref = reference_img[:, :, ch].ravel()

        src_hist, _ = np.histogram(src, bins=256, range=(0, 256))
        ref_hist, _ = np.histogram(ref, bins=256, range=(0, 256))
        src_cdf = np.cumsum(src_hist).astype(np.float64)
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf /= ref_cdf[-1]

        interp_map = np.interp(
            np.linspace(0, 1, 256),
            ref_cdf,
            np.arange(256)
        )
        interp_map = np.round(interp_map).astype(np.uint8)
        
        matched_img[:, :, ch] = interp_map[source_img[:, :, ch]]

    return matched_img
