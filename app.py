import tensorflow as tf
import numpy as np
import os
import random
import traceback
import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image
from keras.utils import custom_object_scope
from sklearn.preprocessing import LabelEncoder

# Flask App Initialization
app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = "static/uploads"
OUTFITS_FOLDER = "static/outfits"
RESULTS_FOLDER = "static/results"

for folder in [UPLOAD_FOLDER, OUTFITS_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class labels (Body types must match training data order)
class_names = ['apple', 'hourglass', 'inverted_triangle', 'pear', 'rectangle']

labels = ['apple', 'hourglass', 'inverted_triangle', 'pear', 'rectangle']
encoder = LabelEncoder()
encoder.fit(labels)
print("Encoded class order:", encoder.classes_)

# Load the trained model using custom_object_scope
class BodyTypeClassifier(tf.keras.Model):
    def __init__(self):
        super(BodyTypeClassifier, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(5)  # Assuming 5 classes

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

try:
    with custom_object_scope({'BodyTypeClassifier': BodyTypeClassifier}):
        model = tf.keras.models.load_model("body_type_model.keras")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# MediaPipe Pose for body detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to recommend outfits based on body type, occasion, and weather
def recommend_outfits(body_type, occasion, weather):
    # Define basic suggestions for body type, occasion, and weather
    suggestions = {
        'apple': {
            'casual': {
                'spring': 'A flowy dress with a cinched waist.',
                'summer': 'A lightweight shirt with high-waisted shorts.',
                'winter': 'A fitted sweater with dark jeans.',
            },
            'party': {
                'spring': 'A wrap dress with a belt to accentuate your waist.',
                'summer': 'A sleeveless top with a maxi skirt.',
                'winter': 'A stylish coat with a slim-fit dress.',
            },
        },
        'hourglass': {
            'casual': {
                'spring': 'A blouse with a tailored jacket and jeans.',
                'summer': 'A fitted top with a midi skirt.',
                'winter': 'A form-fitting sweater dress.',
            },
            'party': {
                'spring': 'A bodycon dress with a high neckline.',
                'summer': 'A strapless dress with a cinched waist.',
                'winter': 'A fur coat with a pencil skirt.',
            },
        },
        'inverted_triangle': {
            'casual': {
                'spring': 'A soft, flowy blouse with straight-leg pants.',
                'summer': 'A loose top with a pleated skirt.',
                'winter': 'A cozy cardigan with leggings.',
            },
            'party': {
                'spring': 'A shift dress with a statement necklace.',
                'summer': 'A A-line dress with a V-neck.',
                'winter': 'A tailored blazer with a sleek skirt.',
            },
        },
        'pear': {
            'casual': {
                'spring': 'A tunic with leggings and sandals.',
                'summer': 'A flowy tank top with a skirt.',
                'winter': 'A long sweater with skinny jeans.',
            },
            'party': {
                'spring': 'A knee-length dress with an empire waist.',
                'summer': 'A wrap dress with a belt.',
                'winter': 'A long-sleeve dress with a fitted coat.',
            },
        },
        'rectangle': {
            'casual': {
                'spring': 'A button-up shirt with skinny jeans.',
                'summer': 'A simple tank top with a skirt.',
                'winter': 'A long cardigan with denim jeans.',
            },
            'party': {
                'spring': 'A sheath dress with a flattering waist.',
                'summer': 'A structured dress with bold prints.',
                'winter': 'A layered outfit with a tailored jacket.',
            },
        },
    }

    # Log and check if the combination exists
    print(f"Checking combination: Body Type = {body_type}, Occasion = {occasion}, Weather = {weather}")
    
    try:
        outfit_suggestion = suggestions[body_type][occasion][weather]
        return [outfit_suggestion]  # Return a list for easy handling later
    except KeyError:
        print(f"Suggestion not found for: {body_type}, {occasion}, {weather}")
        return ["No outfit suggestion available for this combination."]

# Home page - Upload Image
@app.route('/')
def home():
    return render_template('upload.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)

    # Preprocess image
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    try:
        # Make prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        # Redirect to the occasion_weather page with the body_type parameter
        return redirect(url_for('occasion_weather', body_type=predicted_label))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed."}), 500

# Occasion & Weather Selection
@app.route('/occasion_weather', methods=['GET', 'POST'])
def occasion_weather():
    if request.method == 'GET':
        body_type = request.args.get('body_type', '')
        if not body_type:
            return "Error: Body type not found!", 400

        print("Body Type from URL:", body_type)  # Debugging line
        return render_template('occasion_weather.html', body_type=body_type)

    elif request.method == 'POST':
        body_type = request.form.get('body_type', '')
        occasion = request.form.get('occasion', '')
        weather = request.form.get('weather', '')

        print(f"Received Body Type: {body_type}, Occasion: {occasion}, Weather: {weather}")  # Debugging line

        if not body_type:
            return "Error: Body type is missing!", 400

        # Get the outfit recommendations
        recommended_outfits = recommend_outfits(body_type, occasion, weather)

        # Render the result page with the recommendations
        return render_template('recommend.html', body_type=body_type, recommendations=recommended_outfits)


# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
