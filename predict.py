import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras

from src.config import IMAGE_SIZE, MODEL_PATH


# Load trained model
model = keras.models.load_model(MODEL_PATH)


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(IMAGE_SIZE)

    # Convert to array
    image_array = np.array(image, dtype=np.float32)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    probabilities = model.predict(image_array, verbose=0)

    predicted_class = np.argmax(probabilities[0])
    confidence = np.max(probabilities[0]) * 100

    return predicted_class, confidence


def choose_image():

    file_path = filedialog.askopenfilename(
        title="Select Traffic Sign Image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    # Display image
    image = Image.open(file_path)
    image.thumbnail((300, 300))

    photo = ImageTk.PhotoImage(image)

    image_label.config(image=photo)
    image_label.image = photo

    # Prediction
    predicted_class, confidence = predict_image(file_path)

    result_label.config(
        text=f"Predicted Class: {predicted_class}\n"
             f"Confidence: {confidence:.2f}%"
    )


# =========================
# GUI
# =========================

root = tk.Tk()

root.title("Traffic Sign Classification")
root.geometry("500x600")

title_label = tk.Label(
    root,
    text="Traffic Sign Classification",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=20)


button = tk.Button(
    root,
    text="Choose Image",
    font=("Arial", 14),
    command=choose_image
)

button.pack(pady=10)


image_label = tk.Label(root)

image_label.pack(pady=20)


result_label = tk.Label(
    root,
    text="Select an image to get a prediction",
    font=("Arial", 14)
)

result_label.pack(pady=20)


root.mainloop()