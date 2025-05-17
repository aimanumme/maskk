import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained Keras model
MODEL_PATH = 'mask_classifier_model.h5'
model = load_model(MODEL_PATH)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route serving the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img = Image.open(file.stream).convert('RGB')
        # Resize image to model's expected input size
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize if model expects
        # Predict
        preds = model.predict(img_array)
        # Assuming binary classification: 0 = Mask, 1 = No Mask
        pred_class = np.argmax(preds, axis=1)[0] if preds.shape[-1] > 1 else int(preds[0][0] > 0.5)
        if pred_class == 0:
            result = 'Mask Detected 😷'
        else:
            result = 'No Mask 😐'
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 