from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from huggingface_hub import from_pretrained_keras
import tensorflow as tf 
import numpy as np

from sentence_transformers import SentenceTransformer
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ROWS, COLS = 150, 150
model_image = from_pretrained_keras("cats_vs_dogs")
embedding = SentenceTransformer(r"C:\Users\mmumt\Downloads\b7bot\multilingual-e5-large").to('cuda')
model_text = tf.keras.models.load_model('nn_sentiment.h5')

def predict_sentiment(text):
    encoded_text = embedding.encode(text).reshape(1, 1024)
    pred = np.argmax(model_text.predict(encoded_text))
    if pred == 0:
        return 'negative'
    elif pred == 1:
        return 'neutral'
    elif pred == 2:
        return 'positive'
    return pred

def predict_image(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img = img / 255.0
    img = img.reshape(1, ROWS, COLS, 3)

    prediction = model_image.predict(img)[0][0]
    if prediction >= 0.5:
        return {'class': 'Cat', 'confidence': f"{prediction:.2%}"}
    else:
        return {'class': 'Dog', 'confidence': f"{1-prediction:.2%}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    prediction = predict_image(file_path)

    os.remove(file_path)  
    return jsonify(prediction)

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment_endpoint():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    sentiment = predict_sentiment(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
