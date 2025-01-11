from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from PIL import Image
import io
from flask_cors import CORS


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins or specify specific origins


# URL of the model
url = "https://ramapp.dev/fruit_quality_classifier.h5"

# Download the model file
response = requests.get(url)
with open("fruit_quality_classifier.h5", "wb") as f:
    f.write(response.content)

# Load the model
model = load_model("fruit_quality_classifier.h5")
print("Model loaded successfully.")

# Define image size (same as used during training)
image_size = (192, 256)
class_names = ["Bad", "Good", "Mixed"]

# API URL and key for market rates
api_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
api_key = "579b464db66ec23bdd0000016fbfcefc53914cc969e972033b8e2bee"

# Function to fetch rates from API
def fetch_rates():
    try:
        print("Fetching rates... from govt api")
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": 1
        }
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        rates = []
        for record in data.get("records", []):
            min_price = float(record.get("min_price", 0))
            max_price = float(record.get("max_price", 0))
            rates.append((min_price, max_price))
        return rates if rates else []
    except Exception as e:
        print(f"Error fetching rates: {e}")
        return []

# Function to get the appropriate rate based on quality
def get_rate(quality):
    rates = fetch_rates()
    if quality == "Good":
        return f"Current Market Price (approximate): {max(rates, key=lambda x: x[1])[1]}"
    elif quality == "Mixed":
        return f"Current Market Price (approximate): {min(rates, key=lambda x: x[0])[0]}"
    else:  # Bad
        mixed_price = min(rates, key=lambda x: x[0])[0]
        return f"Current Market Price (approximate): {int(mixed_price / 3)}"

# Function to predict fruit quality
def predict_fruit_quality(img):
    img = img.resize(image_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# API endpoint for uploading image and getting prediction.
@app.route("/", methods=["GET"])
def index():
    return "Server is Running Successfully"
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Read the image file
        img = Image.open(io.BytesIO(file.read()))

        # Predict the quality
        quality = predict_fruit_quality(img)
        rate = get_rate(quality)

        return jsonify({"quality": quality, "rate": rate}), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)