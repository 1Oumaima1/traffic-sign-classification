import cv2
import numpy as np

from pathlib import Path

from src.config import IMAGE_SIZE


def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE)

    image = image.astype(np.float32) / 255.0

    return image