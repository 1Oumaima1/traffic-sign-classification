import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.preprocessing import load_image


DATA_DIR = Path("data/dataset")


def load_train_data():
    csv_path = DATA_DIR / "train.csv"

    df = pd.read_csv(csv_path)

    return df


def load_test_data():
    csv_path = DATA_DIR / "test.csv"

    df = pd.read_csv(csv_path)

    return df


def prepare_train_data(validation_size=0.2):
    df = load_train_data()

    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = DATA_DIR / row["Path"]

        image = load_image(image_path)

        images.append(image)
        labels.append(row["ClassId"])

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val

def prepare_test_data():
    df = load_test_data()

    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = DATA_DIR / row["Path"]

        image = load_image(image_path)

        images.append(image)
        labels.append(row["ClassId"])

    X_test = np.array(images, dtype=np.float32)
    y_test = np.array(labels, dtype=np.int64)

    return X_test, y_test