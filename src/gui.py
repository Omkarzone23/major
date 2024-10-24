import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from src.image_loader import load_image
from src.image_filters import apply_grayscale, apply_median_filter, apply_high_pass_filter
from src.segmentation import apply_watershed_segmentation
from src.morphological_refinements import apply_morphological_operations
from src.tumor_properties import (
    calculate_tumor_area, 
    calculate_tumor_perimeter, 
    plot_tumor_boundary,
    locate_tumor_area
)

class BrainTumorDetectionGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Brain Tumor Detection")
        
        self.original_image = None
        self.processed_image = None

        # Buttons
        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.grayscale_button = tk.Button(master, text="Grayscale Image", command=self.grayscale_image)
        self.grayscale_button.pack()

        self.median_filter_button = tk.Button(master, text="Median Filter", command=self.apply_median_filter)
        self.median_filter_button.pack()

        self.high_pass_filter_button = tk.Button(master, text="High Pass Filter", command=self.apply_high_pass_filter)
        self.high_pass_filter_button.pack()

        self.watershed_button = tk.Button(master, text="Watershed Segmentation", command=self.apply_watershed_segmentation)
        self.watershed_button.pack()

        self.morphological_button = tk.Button(master, text="Morphological Operation", command=self.apply_morphological_operations)
        self.morphological_button.pack()

        self.tumor_area_location_button = tk.Button(master, text="Tumor Area Location", command=self.locate_tumor_area)
        self.tumor_area_location_button.pack()

        self.tumor_boundary_button = tk.Button(master, text="Tumor Boundary", command=self.plot_tumor_boundary)
        self.tumor_boundary_button.pack()

        self.tumor_area_button = tk.Button(master, text="Tumor Area", command=self.calculate_tumor_area)
        self.tumor_area_button.pack()

        self.tumor_perimeter_button = tk.Button(master, text="Tumor Perimeter", command=self.calculate_tumor_perimeter)
        self.tumor_perimeter_button.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = load_image(file_path)
            self.processed_image = self.original_image.copy()
            self.show_image(self.original_image, "Original Image")
            messagebox.showinfo("Image Loaded", "Image has been loaded successfully.")

    def grayscale_image(self):
        if self.original_image is not None:
            self.processed_image = apply_grayscale(self.original_image)
            self.show_image(self.processed_image, "Grayscale Image")
            messagebox.showinfo("Grayscale Applied", "Grayscale conversion completed.")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def apply_median_filter(self):
        if self.original_image is not None:
            self.processed_image = apply_median_filter(self.original_image)
            self.show_image(self.processed_image, "Median Filtered Image")
            messagebox.showinfo("Median Filter Applied", "Median filter completed.")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def apply_high_pass_filter(self):
        if self.original_image is not None:
            self.processed_image = apply_high_pass_filter(self.original_image)
            self.show_image(self.processed_image, "High Pass Filtered Image")
            messagebox.showinfo("High Pass Filter Applied", "High pass filter completed.")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def apply_watershed_segmentation(self):
        if self.original_image is not None:
            self.processed_image = apply_watershed_segmentation(self.original_image)
            self.show_image(self.processed_image, "Watershed Segmented Image")
            messagebox.showinfo("Watershed Segmentation", "Watershed segmentation completed.")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def apply_morphological_operations(self):
        if self.original_image is not None:
            self.processed_image = apply_morphological_operations(self.original_image)
            self.show_image(self.processed_image, "Morphological Operations Image")
            messagebox.showinfo("Morphological Operations", "Morphological operations completed.")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def locate_tumor_area(self):
        if self.original_image is not None:
            boxed_image = locate_tumor_area(self.processed_image)
            self.show_image(boxed_image, "Tumor Area Located")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def plot_tumor_boundary(self):
        if self.processed_image is not None:
            boundary_image = plot_tumor_boundary(self.processed_image)
            self.show_image(boundary_image, "Tumor Boundary Image")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def calculate_tumor_area(self):
        if self.processed_image is not None:
            area = calculate_tumor_area(self.processed_image)
            messagebox.showinfo("Tumor Area", f"Tumor area: {area:.2f} pixels")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def calculate_tumor_perimeter(self):
        if self.processed_image is not None:
            perimeter = calculate_tumor_perimeter(self.processed_image)
            messagebox.showinfo("Tumor Perimeter", f"Tumor perimeter: {perimeter:.2f} pixels")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def show_image(self, image, title):
        # Convert the image to a format suitable for displaying in tkinter
        image_resized = cv2.resize(image, (500, 500))  # Resize if the image is too big
        cv2.imshow(title, image_resized)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close all OpenCV windows when done

def run_gui():
    root = tk.Tk()
    app = BrainTumorDetectionGUI(root)
    root.mainloop()   