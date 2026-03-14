import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Dataset path
dataset_path = "leapGestRecog"


# Image size
img_size = 64


# Lists for data and labels
X = []
y = []


# Load dataset
for subject in os.listdir(dataset_path):

    subject_path = os.path.join(dataset_path, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):

        gesture_path = os.path.join(subject_path, gesture)

        for img in os.listdir(gesture_path)[:500]:

            img_path = os.path.join(gesture_path, img)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            image = cv2.resize(image, (img_size, img_size))
            image = image.flatten()

            X.append(image)
            y.append(gesture)


# Convert to numpy arrays
X = np.array(X)
y = np.array(y)


print("Dataset shape:", X.shape)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create SVM model
model = SVC(kernel="linear")


# Train model
model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)