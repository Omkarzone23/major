import cv2

def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return cv2.resize(image, (256, 256))
