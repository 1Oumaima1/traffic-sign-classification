import numpy as np
import pandas as pd

from pathlib import Path
from tensorflow import keras
from sklearn.metrics import classification_report

from src.data_loader import prepare_test_data
from src.config import MODEL_PATH, NUM_CLASSES


OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_report():

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

    # Generate classification report
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )

    # Display report
    print("\n========================================")
    print("       CLASSIFICATION REPORT")
    print("========================================\n")

    print(report)

    # Save report as text file
    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report)

    print("\nSaved:", report_path)

    # Save report as CSV
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    csv_path = OUTPUT_DIR / "classification_report.csv"

    report_df.to_csv(csv_path)

    print("Saved:", csv_path)

    print("\nClassification report completed successfully!")


if __name__ == "__main__":
    generate_report()