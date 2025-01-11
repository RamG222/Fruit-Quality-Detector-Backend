import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation

# Step 0: Enable GPU Support
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"Error enabling GPU: {e}")
else:
    print("No GPU detected, using CPU.")

# Step 1: Load the Dataset
dataset_dir = "./dataset"  # Relative path to the dataset
image_size = (192, 256)  # Image dimensions
batch_size = 32  # Number of images per batch

# Load dataset
dataset = image_dataset_from_directory(
    dataset_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'  # Labels are one-hot encoded
)

# Split into training (80%) and validation (20%)
train_dataset = dataset.take(int(0.8 * len(dataset)))
val_dataset = dataset.skip(int(0.8 * len(dataset)))

# Step 2: Data Augmentation and Normalization
# Data augmentation for training data
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    Rescaling(1.0 / 255)  # Normalize pixel values to [0, 1]
])

# Normalize validation data (no augmentation)
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# Apply augmentation to training data
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

# Step 3: Define the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(192, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer for 3 classes: Good, Bad, Mixed
])

# Step 4: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Step 6: Save the Model
model.save("fruit_quality_classifier.h5")
print("Model saved as 'fruit_quality_classifier.h5'")
