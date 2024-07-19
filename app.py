import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load_model('models/peonymodelv4.h5')

# Serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Decode the image and preprocess it
def preprocess_image(image_data):
    image = Image.open(BytesIO(base64.b64decode(image_data.split(',')[1])))
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((256, 256))  
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Prediction endpoint
@app.route('/identify', methods=['POST'])
def identify():
    data = request.get_json()
    image_data = data['image']
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    predicted_class = 'Peony' if prediction[0] > 0.5 else 'Not Peony'
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
