from tensorflow import keras

from src.data_loader import prepare_test_data
from src.config import MODEL_PATH


def evaluate_model():
    print("Loading test dataset...")

    X_test, y_test = prepare_test_data()

    print("Test samples:", len(X_test))

    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\nTest Loss:", loss)
    print("Test Accuracy:", accuracy)


if __name__ == "__main__":
    evaluate_model()