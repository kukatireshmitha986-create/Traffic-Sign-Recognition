import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# 1. Page Configuration
# ==========================================

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)


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
# 3. Load Model
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/traffic_sign_cnn.keras"
    )


model = load_model()


# ==========================================
# 4. Header
# ==========================================

st.title("🚦 Traffic Sign Recognition")

st.write(
    "Upload a traffic sign image and the CNN model "
    "will predict the traffic sign."
)

st.divider()


# ==========================================
# 5. Image Upload
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg", "jpeg", "png", "ppm"]
)


# ==========================================
# 6. Prediction
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded Traffic Sign",
        width=300
    )

    st.write("")

    if st.button(
        "🔍 Predict Traffic Sign",
        use_container_width=True
    ):

        # Convert image to RGB
        image = image.convert("RGB")

        # Resize image
        image = image.resize(
            (32, 32)
        )

        # Convert to NumPy
        image_array = np.array(
            image
        )

        # Normalize
        image_array = image_array.astype(
            "float32"
        ) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            image_array,
            verbose=0
        )

        # Get class
        predicted_class = np.argmax(
            prediction[0]
        )

        # Get confidence
        confidence = np.max(
            prediction[0]
        ) * 100

        # ==================================
        # Display Result
        # ==================================

        st.success(
            "Prediction completed!"
        )

        st.subheader(
            "Prediction Result"
        )

        st.write(
            f"### 🚦 {CLASS_NAMES[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            f"Predicted Class: {predicted_class}"
        )


# ==========================================
# 7. Project Information
# ==========================================

st.divider()

st.caption(
    "Traffic Sign Recognition using Convolutional Neural Network (CNN)"
)

st.caption(
    "Dataset: German Traffic Sign Recognition Benchmark (GTSRB)"
)