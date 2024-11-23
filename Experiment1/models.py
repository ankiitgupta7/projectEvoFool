import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

def get_model(model_name, input_shape):
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
    if model_name in ["CNN", "RNN"]:
        model.fit(X_train[..., np.newaxis], y_train, epochs=10, batch_size=32, verbose=0)
    else:
        model.fit(X_train.reshape((-1, X_train.shape[1] * X_train.shape[2])), y_train.argmax(axis=1))
    return model
