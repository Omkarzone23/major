import cv2
import numpy as np

def apply_watershed_segmentation(image):
    """Apply watershed segmentation to the image."""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding after Gaussian filtering
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to remove small noise and fill small holes
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Dilation to get sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform to get sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Convert sure foreground to uint8 type
    sure_fg = np.uint8(sure_fg)

    # Find the unknown region (area between background and foreground)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Label the markers
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all the markers so that the sure background is not 0, but 1
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply the watershed algorithm
    cv2.watershed(image, markers)
    
    # Mark boundaries with red where marker is -1 (the boundaries detected by watershed)
    image[markers == -1] = [0, 0, 255]  # Red boundaries for segmentation result

    # Optionally, overlay the segmented regions with color
    image[markers > 1] = [0, 255, 0]  # Color segmented areas green (optional)

    # Display the segmented image
    cv2.imshow('Watershed Segmentation', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image