from tensorflow import keras

from src.data_loader import prepare_train_data
from src.model import build_model
from src.config import EPOCHS, MODEL_PATH


def train_model():
    print("Loading dataset...")

    X_train, X_val, y_train, y_val = prepare_train_data()

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    model = build_model()

    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[
            checkpoint,
            early_stopping
        ]
    )

    return model, history


if __name__ == "__main__":
    train_model()