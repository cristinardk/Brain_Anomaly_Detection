import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix

# Define the dataset path
dataset_path = "dataset"

# Define training and testing directories
train_dir = os.path.join(dataset_path, "Training")
test_dir = os.path.join(dataset_path, "Testing")

# Define the categories
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Function to visualize the distribution of tumor types
def visualize_distribution(train_dir, categories):
    train_data = []
    for category in categories:
        folder_path = os.path.join(train_dir, category)
        images = os.listdir(folder_path)
        count = len(images)
        train_data.append(pd.DataFrame({"Image": images, "Category": [category] * count, "Count": [count] * count}))
    train_df = pd.concat(train_data, ignore_index=True)

    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.barplot(data=train_df, x="Category", y="Count")
    plt.title("Distribution of Tumor Types")
    plt.xlabel("Tumor Type")
    plt.ylabel("Count")
    plt.show()

# Function to visualize sample images
def visualize_sample_images(train_dir, categories):
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        folder_path = os.path.join(train_dir, category)
        image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        img = plt.imread(image_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(category)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Set image size, batch size, and epochs
image_size = (128, 128)
batch_size = 32
epochs = 50

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Define the CNN model
def build_model(image_size, categories):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(len(categories), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train the model
model = build_model(image_size, categories)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Plot training history
def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Generate predictions and plot confusion matrix
def plot_confusion_matrix(test_generator, model, categories):
    predictions = model.predict(test_generator)
    predicted_categories = np.argmax(predictions, axis=1)
    true_categories = test_generator.classes

    cm = confusion_matrix(true_categories, predicted_categories)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(categories)), labels=categories)
    plt.yticks(ticks=np.arange(len(categories)), labels=categories)
    plt.show()

    print("Classification Report:")
    print(classification_report(true_categories, predicted_categories, target_names=categories))

plot_confusion_matrix(test_generator, model, categories)

# Save the model
model.save("brain_tumor_detection_model.h5")
