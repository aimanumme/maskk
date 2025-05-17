import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf

# === Initialize Flask app ===
app = Flask(__name__)

# === Get the absolute path to the model file ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mask_classifier_model.h5")

# === Load the model ===
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# === Allowed file extensions ===
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Home route ===
@app.route('/')
def index():
    return render_template('index.html')

# === Prediction route ===
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Read and process image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0] if preds.shape[-1] > 1 else int(preds[0][0] > 0.5)
        result = 'Mask Detected 😷' if pred_class == 0 else 'No Mask 😐'
        
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# === Run locally ===
if __name__ == '__main__':
    app.run(debug=True)
