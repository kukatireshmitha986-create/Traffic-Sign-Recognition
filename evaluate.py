import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# ==========================================
# 1. Settings
# ==========================================

DATASET_PATH = "dataset/Train"
MODEL_PATH = "models/traffic_sign_cnn.keras"

IMG_SIZE = 32
NUM_CLASSES = 43

images = []
labels = []


# ==========================================
# 2. Load Dataset
# ==========================================

print("Loading dataset for evaluation...")

for class_id in range(NUM_CLASSES):

    class_path = os.path.join(
        DATASET_PATH,
        str(class_id)
    )

    if not os.path.exists(class_path):
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(
            class_path,
            image_name
        )

        try:

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            images.append(image)
            labels.append(class_id)

        except Exception:
            continue


# ==========================================
# 3. Convert Data
# ==========================================

X = np.array(images)
y = np.array(labels)

X = X.astype("float32") / 255.0

print("\nDataset loaded successfully!")
print("Total images:", len(X))


# ==========================================
# 4. Create Same Test Split
# ==========================================

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Testing images:", len(X_test))


# ==========================================
# 5. Load Trained Model
# ==========================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# 6. Make Predictions
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# 7. Classification Report
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    y_test,
    y_pred
)

print(report)


# ==========================================
# 8. Save Classification Report
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ==========================================
# 9. Confusion Matrix
# ==========================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    y_pred
)


# ==========================================
# 10. Save Confusion Matrix
# ==========================================

fig, ax = plt.subplots(
    figsize=(14, 12)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

plt.title(
    "Traffic Sign Recognition - Confusion Matrix"
)

plt.savefig(
    "results/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 11. Final Message
# ==========================================

print("\n================================")
print("EVALUATION COMPLETED")
print("================================")

print("Classification report:")
print("results/classification_report.txt")

print("\nConfusion matrix:")
print("results/confusion_matrix.png")