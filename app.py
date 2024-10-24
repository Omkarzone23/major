from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import cv2
import numpy as np
import os
from io import BytesIO
from flask_cors import CORS


# Import your existing image processing functions
from src.image_loader import load_image
from src.image_filters import apply_grayscale, apply_median_filter, apply_high_pass_filter
from src.segmentation import apply_watershed_segmentation
from src.morphological_refinements import apply_morphological_operations
from src.seg import greyscale, watershed
from src.tumor_properties import calculate_tumor_area, calculate_tumor_perimeter, plot_tumor_boundary, locate_tumor_area

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)
# Set the upload folder for static files
UPLOAD_FOLDER = 'static\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['image']
        if file:
          
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.jpg')
            print(file_path)
            file.save(file_path)

            # Load the image using your existing function
            original_image = load_image(file_path)

            # Apply the selected processing technique
            operation = request.form.get('operation')
            processed_image = original_image

            if operation == 'grayscale':
                processed_image = greyscale(original_image)
            elif operation == 'median_filter':
                processed_image = apply_median_filter(original_image)
            elif operation == 'high_pass_filter':
                processed_image = apply_high_pass_filter(original_image)
            elif operation == 'watershed_segmentation':
                processed_image = watershed(original_image)
            elif operation == 'morphological_operations':
                processed_image = apply_morphological_operations(original_image)
            elif operation == 'locate_tumor_area':
                processed_image = locate_tumor_area(original_image)
            elif operation == 'plot_tumor_boundary':
                processed_image = plot_tumor_boundary(original_image)

            #save different
            save_output(processed_image, image_name=operation)
            # Save the processed image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
            cv2.imwrite(output_path, processed_image)

            # Display additional properties if applicable
            return "done"
    return render_template("index.html")
        
@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_image.jpg')
        file.save(file_path)

        # Load the image using your existing function
        original_image = load_image(file_path)

        # Apply the selected processing technique
        operation = request.form.get('operation')
        processed_image = original_image

        if operation == 'grayscale':
            processed_image = greyscale(original_image)
        elif operation == 'median_filter':
            processed_image = apply_median_filter(original_image)
        elif operation == 'high_pass_filter':
            processed_image = apply_high_pass_filter(original_image)
        elif operation == 'watershed_segmentation':
            processed_image = watershed(original_image)
        elif operation == 'morphological_operations':
            processed_image = apply_morphological_operations(original_image)
        elif operation == 'locate_tumor_area':
            processed_image = locate_tumor_area(original_image)
        elif operation == 'plot_tumor_boundary':
            processed_image = plot_tumor_boundary(original_image)

        # Convert the processed image to a format that can be sent as a response
        _, buffer = cv2.imencode('.jpg', processed_image)
        io_buf = BytesIO(buffer)

        # Send the image as a file response
        return send_file(io_buf, mimetype='image/jpeg')

def save_output(processed_image, image_name):
            # Save the processed image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
            cv2.imwrite(output_path, processed_image)

if __name__ == '__main__':
    app.run(debug=True)
