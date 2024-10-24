import cv2
import numpy as np

def apply_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_median_filter(image):
    """Apply median filter to the image."""
    return cv2.medianBlur(image, 5)

def apply_high_pass_filter(image):
    """Apply a high-pass filter to the image."""
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)
