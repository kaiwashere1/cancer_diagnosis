from flask import Flask, request, render_template_string, send_from_directory, jsonify
from keras.models import load_model
import numpy as np
import os
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# HTML Template for rendering the web page
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="icon" type="image/x-icon" href="{{ url_for('static', filename='favicon.ico') }}">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Diagnosis App</title>
    <style>
        /* Styles omitted for brevity */
    </style>
</head>
<body>
    <header>
        <img src="{{ url_for('static', filename='oncovision.png') }}" alt="Logo" class="logo">
        <h1>Cancer Diagnosis App</h1>
    </header>

    <main>
        <div class="form-container">
            <h2>Upload an Image for Diagnosis</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <label for="file">Select Image File:</label>
                <input type="file" id="file" name="file" required>
                <button type="submit">Predict</button>
            </form>

            {% if prediction %}
            <div class="result">
                <h3>Prediction Result:</h3>
                <p>{{ prediction }}</p>
            </div>
            {% endif %}
        </div>
    </main>

    <footer>
        &copy; 2024 Oncovision.ai, All rights reserved.
    </footer>
</body>
</html>
'''

# Load your models (ensure these paths are correct)
model1 = load_model('models/breast_cancer_model.keras')
model2 = load_model('models/lung_cancer_model.keras')
model3 = load_model('models/skin_cancer_model.keras')


def preprocess_file(file):
    """Preprocess the uploaded image file."""
    img = Image.open(file)
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if not file:
        return jsonify({'error': 'No file provided'}), 400

    try:
        # Preprocess the file
        img_array = preprocess_file(file)

        # Make predictions
        prediction1 = model1.predict(img_array)[0][0]
        prediction2 = model2.predict(img_array)[0][0]
        prediction3 = model3.predict(img_array)[0][0]

        # Format the prediction results
        prediction_result = f"Breast Cancer Model: {prediction1:.2f}, Lung Cancer Model: {prediction2:.2f}, Skin Cancer Model: {prediction3:.2f}"

        return render_template_string(HTML_TEMPLATE, prediction=prediction_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(debug=True)
