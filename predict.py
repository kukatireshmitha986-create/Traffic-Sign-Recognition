import os
import cv2
import numpy as np
import tensorflow as tf


# ==========================================
# 1. Model Settings
# ==========================================

MODEL_PATH = "models/traffic_sign_cnn.keras"

IMG_SIZE = 32


# ==========================================
# 2. Traffic Sign Class Names
# ==========================================

CLASS_NAMES = [
    "Speed limit 20 km/h",
    "Speed limit 30 km/h",
    "Speed limit 50 km/h",
    "Speed limit 60 km/h",
    "Speed limit 70 km/h",
    "Speed limit 80 km/h",
    "End of speed limit 80 km/h",
    "Speed limit 100 km/h",
    "Speed limit 120 km/h",
    "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]


# ==========================================
# 3. Load Trained Model
# ==========================================

print("Loading trained CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# 4. Get Image Path
# ==========================================

image_path = input(
    "\nEnter the path of the traffic sign image: "
).strip()


# Remove quotes if user pastes a quoted path
image_path = image_path.strip('"')


# ==========================================
# 5. Check Image
# ==========================================

if not os.path.exists(image_path):

    print("\nERROR: Image not found!")

    print("Please check the image path.")

    exit()


# ==========================================
# 6. Load Image
# ==========================================

image = cv2.imread(
    image_path
)

if image is None:

    print("\nERROR: Could not read the image.")

    exit()


# ==========================================
# 7. Preprocess Image
# ==========================================

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

image = cv2.resize(
    image,
    (IMG_SIZE, IMG_SIZE)
)

image = image.astype(
    "float32"
) / 255.0


# Add batch dimension

image = np.expand_dims(
    image,
    axis=0
)


# ==========================================
# 8. Prediction
# ==========================================

print("\nPredicting traffic sign...")

prediction = model.predict(
    image,
    verbose=0
)


# ==========================================
# 9. Get Prediction
# ==========================================

predicted_class = np.argmax(
    prediction[0]
)

confidence = np.max(
    prediction[0]
) * 100


# ==========================================
# 10. Display Result
# ==========================================

print("\n========================================")
print("TRAFFIC SIGN PREDICTION")
print("========================================")

print(
    "Predicted Class:",
    predicted_class
)

print(
    "Traffic Sign:",
    CLASS_NAMES[predicted_class]
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("========================================")