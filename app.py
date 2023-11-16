import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model  # Use tf.keras instead of just keras
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Load the binary classification model (tumor yes or no)
binary_model = load_model('BrainTumor10EpochsCategorical.h5')

# Load the multi-class tumor classification model
multi_model = load_model('MultiClassTumorModel.h5')

print('Models loaded. Check http://127.0.0.1:5000/')

def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"

def get_tumor_type_name(tumor_type):
    if tumor_type == 0:
        return "Glioma"
    elif tumor_type == 1:
        return "Meningioma"
    elif tumor_type == 2:
        return "Pituitary"

def get_result(img_path):
    image = cv2.imread(img_path)
    print("File Path:", img_path)
    if image is not None:
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)

        # Use the binary model to check if a tumor is detected
        binary_result = binary_model.predict(input_img)

        if binary_result[0][1] > 0.5:
            # If a tumor is detected, determine the tumor type using the multi-class model
            multi_result = multi_model.predict(input_img)
            tumor_type = np.argmax(multi_result)
            return f"Tumor Detected: {get_tumor_type_name(tumor_type)}"
        else:
            return "No Tumor Detected"
    else:
        return "Error: Image not found or couldn't be loaded"


def get_tumor_type(img_path):
    image = cv2.imread(img_path)
    print("File Path:", img_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)

    binary_result = binary_model.predict(input_img)

    if binary_result[0][1] > 0.5:
        multi_result = multi_model.predict(input_img)
        tumor_type = np.argmax(multi_result)
        return get_tumor_type_name(tumor_type)
    else:
        return "No Tumor Detected"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        result = get_result(file_path)
        return result
    return "No file uploaded"

@app.route('/get_tumor_type', methods=['POST'])
def get_tumor_type_endpoint():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        result = get_tumor_type(file_path)
        return result
    return "No file uploaded"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
