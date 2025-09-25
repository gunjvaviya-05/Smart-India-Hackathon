import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = 'dataset'
CATEGORIES = ['healthy', 'infected']

def load_dataset():
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        class_num = CATEGORIES.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = load_img(img_path, target_size=IMAGE_SIZE)
                img_array = img_to_array(img) / 255.0
                data.append(img_array)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image: {img_path}, {e}")

    return np.array(data), np.array(labels)

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model():
    data, labels = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    model = build_model()
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        validation_data=(X_val, y_val),
                        batch_size=BATCH_SIZE)

    model.save('plant_disease_model.h5')
    print("✅ Model trained and saved.")
    return model

def predict_image(model_path, image_path):
    from tensorflow.keras.models import load_model

    model = load_model(model_path)
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        print("🌿 The plant is INFECTED.")
    else:
        print("✅ The plant is HEALTHY.")

# Main function
if __name__ == '__main__':
    import sys

    if not os.path.exists("plant_disease_model.h5"):
        model = train_model()
    else:
        print("📦 Using existing trained model.")
        from tensorflow.keras.models import load_model
        model = load_model("plant_disease_model.h5")

    # Provide your test image path
    test_image_path = "predict_image.jpg"
    if os.path.exists(test_image_path):
        predict_image("plant_disease_model.h5", test_image_path)
    else:
        print(f"❌ Test image '{test_image_path}' not found.")
