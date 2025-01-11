import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Step 1: Load the Pre-trained Model
model = load_model("fruit_quality_classifier.h5")
print("Model loaded successfully.")

# Step 2: Define the Image Size (same as used during training)
image_size = (192, 256)  # Image dimensions

# Step 3: Test the Model with a New Image
def predict_fruit_quality(image_path):
    # Load and preprocess the test image
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    class_names = [ "Bad","Good", "Mixed"]
    predicted_class = class_names[np.argmax(predictions)]

    print(f"Predicted Quality: {predicted_class}")
    return predicted_class

# Example: Predict quality of test.jpg
test_image_path = "test.jpg"  # Relative path to the test image
predict_fruit_quality(test_image_path)
