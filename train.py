import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


# ==========================================
# 1. Dataset Path
# ==========================================

DATASET_PATH = "dataset/Train"

# ==========================================
# 2. Image Settings
# ==========================================

IMG_SIZE = 32
NUM_CLASSES = 43

images = []
labels = []


# ==========================================
# 3. Load Dataset
# ==========================================

print("Loading dataset...")

for class_id in range(NUM_CLASSES):

    class_path = os.path.join(DATASET_PATH, str(class_id))

    if not os.path.exists(class_path):
        print(f"Class {class_id} not found")
        continue

    for image_name in os.listdir(class_path):

        image_path = os.path.join(class_path, image_name)

        try:
            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            images.append(image)
            labels.append(class_id)

        except Exception as e:
            print("Error loading:", image_path)
            print(e)


# ==========================================
# 4. Convert to NumPy Arrays
# ==========================================

X = np.array(images)
y = np.array(labels)

print("\nDataset loaded successfully!")
print("Total images:", len(X))
print("Image shape:", X.shape)
print("Labels shape:", y.shape)


# ==========================================
# 5. Normalize Images
# ==========================================

X = X.astype("float32") / 255.0

print("\nImages normalized successfully!")


# ==========================================
# 6. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset split completed!")
print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# ==========================================
# 7. Build CNN Model
# ==========================================

print("\nBuilding CNN model...")

model = models.Sequential([
    
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==========================================
# 8. Display Model
# ==========================================

print("\nCNN Model Summary:")

model.summary()


# ==========================================
# 9. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nCNN model compiled successfully!")


# ==========================================
# 10. Train Model
# ==========================================

print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


# ==========================================
# 11. Evaluate Model
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# 12. Save Model
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/traffic_sign_cnn.keras"
)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print("Saved to:")
print("models/traffic_sign_cnn.keras")


# ==========================================
# 13. Create Results Folder
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)


# ==========================================
# 14. Accuracy Graph
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/accuracy.png"
)

plt.close()


# ==========================================
# 15. Loss Graph
# ==========================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/loss.png"
)

plt.close()


# ==========================================
# 16. Final Result
# ==========================================

print("\n================================")
print("PROJECT TRAINING COMPLETED")
print("================================")

print("Model saved at:")
print("models/traffic_sign_cnn.keras")

print("\nGraphs saved at:")
print("results/accuracy.png")
print("results/loss.png")

print("\nFinal Test Accuracy:")
print(f"{test_accuracy * 100:.2f}%")