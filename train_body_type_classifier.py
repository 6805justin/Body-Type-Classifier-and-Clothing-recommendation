import tensorflow as tf
import os
import numpy as np
import random

# Define image parameters
IMG_SIZE = (224, 224)  # Resize all images to this size
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = "body_types"  # Folder where images are stored

# Get class names from folder names
class_names = sorted(os.listdir(DATASET_PATH))
num_classes = len(class_names)
print("Classes:", class_names)

# Function to load and preprocess an image
def load_image(image_path, label):
    # Read the image file
    img = tf.io.read_file(image_path)
    
    # Decode the image based on file extension (JPEG or PNG)
    image_ext = tf.strings.lower(tf.strings.split(image_path, '.')[-1])
    img = tf.cond(
        tf.math.reduce_any(tf.strings.regex_full_match(image_ext, "jpg|jpeg")),
        lambda: tf.image.decode_jpeg(img, channels=3),
        lambda: tf.cond(
            tf.math.reduce_any(tf.strings.regex_full_match(image_ext, "png")),
            lambda: tf.image.decode_png(img, channels=3),
            lambda: tf.zeros([IMG_SIZE[0], IMG_SIZE[1], 3], dtype=tf.uint8)  # Return a blank image if unsupported format
        )
    )
    
    # Resize and normalize the image
    img = tf.image.resize(img, IMG_SIZE)  # Resize to 224x224
    img = img / 255.0  # Normalize pixel values
    return img, label

# Prepare dataset
image_paths = []
labels = []

for label, class_name in enumerate(class_names):
    class_folder = os.path.join(DATASET_PATH, class_name)
    for image_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, image_name)
        # Check for valid image file extensions before adding to dataset
        if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(image_path)
            labels.append(label)

# Shuffle data
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# Convert to TensorFlow dataset
image_paths = list(image_paths)
labels = np.array(labels)

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image).shuffle(len(image_paths)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Split into training, validation, and test sets (60% train, 20% validation, 20% test)
train_size = int(0.6 * len(image_paths))
val_size = int(0.2 * len(image_paths))
test_size = len(image_paths) - train_size - val_size

train_paths, val_paths, test_paths = image_paths[:train_size], image_paths[train_size:train_size+val_size], image_paths[train_size+val_size:]
train_labels, val_labels, test_labels = labels[:train_size], labels[train_size:train_size+val_size], labels[train_size+val_size:]

# Convert the test set to a TensorFlow dataset
test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
test_ds = test_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Define a CNN model using Keras
class BodyTypeClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes)  # 5 output classes

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

# Create model
model = BodyTypeClassifier()

# Compile the model
model.compile(optimizer='adam', 
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

# Train the model
model.fit(dataset.take(train_size), validation_data=dataset.skip(train_size), epochs=EPOCHS)

# Save model
model.save("body_type_model.keras")
print("Model saved successfully!")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
print(model.summary())
