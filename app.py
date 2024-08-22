from flask import Flask, request, render_template_string, send_from_directory
from keras.models import load_model
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="icon" type="image/x-icon" href="{{ url_for('static', filename='favicon.ico') }}">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Diagnosis App</title>
    <style>
        /* Global Styles */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
        }

        body {
            background-color: #f0f2f5;
            color: #333;
        }

        header {
            background-color: #4CAF50;
            color: white;
            padding: 15px 0;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            letter-spacing: 1px;
        }

        .logo {
            max-width: 150px;
            height: auto;
            margin-bottom: 10px;
        }

        main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 30px;
            min-height: 80vh;
        }

        .form-container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 500px;
            text-align: center;
        }

        .form-container h2 {
            margin-bottom: 20px;
            font-size: 28px;
            color: #4CAF50;
        }

        .form-container label {
            font-size: 16px;
            color: #555;
            margin-bottom: 8px;
            display: block;
            text-align: left;
        }

        .form-container input[type="file"] {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            width: 100%;
            margin-bottom: 20px;
            outline: none;
        }

        .form-container button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 15px;
            width: 100%;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .form-container button:hover {
            background-color: #45a049;
        }

        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f8f8;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
        }

        footer {
            background-color: #333;
            color: white;
            padding: 15px;
            text-align: center;
            font-size: 14px;
            position: relative;
            bottom: 0;
            width: 100%;
        }

        @media (max-width: 768px) {
            .form-container {
                width: 90%;
            }
        }
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

# Load your models
model1 = load_model('breast_cancer_model.keras')
model2 = load_model('lung_cancer_model.keras')
model3 = load_model('skin_cancer_model.keras')


def preprocess_file(file):
    # Implement this function to preprocess the file as needed
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

    # Preprocess the file
    img_array = preprocess_file(file)

    # Make predictions
    prediction1 = model1.predict(img_array)[0][0]
    prediction2 = model2.predict(img_array)[0][0]
    prediction3 = model3.predict(img_array)[0][0]

    # Return results
    prediction_result = f"Breast Cancer Model: {prediction1:.2f}, Lung Cancer Model: {prediction2:.2f}, Skin Cancer Model: {prediction3:.2f}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction_result)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')
import requests
url = 'http://127.0.0.1:5000/predict'
data = [1, 2, 3, 4]  # Replace with your actual input data
response = requests.post(url, json=data)
print(response.json())
import tensorflow as tf

# Define the correct path to your model files
model1 = tf.keras.models.load_model('models/breast_cancer_model.keras')
model2 = tf.keras.models.load_model('models/lung_cancer_model.keras')
model3 = tf.keras.models.load_model('models/skin_cancer_model.keras')


if __name__ == '__main__':
    app.run(debug=True)
