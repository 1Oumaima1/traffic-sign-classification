import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data_loader import prepare_test_data
from src.config import MODEL_PATH, NUM_CLASSES


OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def visualize_test_results():

    print("Loading test dataset...")
    X_test, y_test = prepare_test_data()

    print("Test samples:", len(X_test))

    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Making predictions...")

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    y_pred = np.argmax(probabilities, axis=1)

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================

    print("\nCreating confusion matrix...")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    fig, ax = plt.subplots(figsize=(16, 16))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=np.arange(NUM_CLASSES)
    )

    disp.plot(
        ax=ax,
        xticks_rotation="vertical",
        colorbar=False
    )

    ax.set_title("Confusion Matrix - Traffic Sign Classification")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", confusion_path)

    # =========================================================
    # WRONG PREDICTIONS
    # =========================================================

    print("\nFinding wrong predictions...")

    wrong_indices = np.where(y_pred != y_test)[0]

    print("Wrong predictions:", len(wrong_indices))
    print("Correct predictions:", len(y_test) - len(wrong_indices))

    if len(wrong_indices) == 0:
        print("No wrong predictions found.")
        return

    n = min(12, len(wrong_indices))

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9)
    )

    for ax, idx in zip(axes.flat, wrong_indices[:n]):

        image = X_test[idx]

        if image.max() > 1:
            image = image / 255.0

        ax.imshow(np.clip(image, 0, 1))

        ax.set_title(
            f"True: {y_test[idx]} | Pred: {y_pred[idx]}"
        )

        ax.axis("off")

    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "Wrong Predictions - Traffic Sign Classification",
        fontsize=16
    )

    plt.tight_layout()

    wrong_path = OUTPUT_DIR / "wrong_predictions.png"

    plt.savefig(
        wrong_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", wrong_path)

    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    visualize_test_results()