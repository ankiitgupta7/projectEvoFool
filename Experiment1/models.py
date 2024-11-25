import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

def get_model(model_name, input_shape):
    """
    Returns the appropriate model based on the model_name.
    """
    if model_name == "CNN":
        return create_cnn_model(input_shape)
    elif model_name == "RNN":
        return create_rnn_model(input_shape)
    elif model_name == "SVM":
        return SVC(gamma="scale", probability=True)
    elif model_name == "RF":
        return RandomForestClassifier()
    elif model_name == "GBM":
        return GradientBoostingClassifier()
    elif model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_cnn_model(input_shape):
    """
    Creates and compiles a CNN model.
    """
    # Ensure input_shape includes the channel dimension
    if len(input_shape) == 2:
        input_shape = (*input_shape, 1)  # Add channel dimension for grayscale images

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def create_rnn_model(input_shape):
    """
    Creates and compiles an RNN model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(50, return_sequences=True),
        tf.keras.layers.SimpleRNN(50),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, model_name, X_train, y_train):
    """
    Trains the provided model on the training data.
    Automatically adjusts input shape for CNN and RNN models.
    """
    if model_name in ["CNN", "RNN"]:
        # Add channel dimension for grayscale images
        if len(X_train.shape) == 3:  # Grayscale data
            X_train = X_train[..., np.newaxis]
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    else:
        # Flatten the input for non-CNN/RNN models
        model.fit(X_train.reshape((X_train.shape[0], -1)), y_train.argmax(axis=1))
    return model
