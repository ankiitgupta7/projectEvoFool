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


def evaluate_model_accuracy(model, model_name, X_train, y_train, X_test, y_test):
    """
    Evaluate overall and per-class accuracy of the model.
    """
    print("\nModel Evaluation:")
    
    # Predict and calculate overall accuracy
    if model_name in ["CNN", "RNN"]:
        train_pred = model.predict(X_train[..., np.newaxis])  # 2D predictions
        test_pred = model.predict(X_test[..., np.newaxis])    # 2D predictions
        train_pred = np.argmax(train_pred, axis=1)  # Convert to class indices
        test_pred = np.argmax(test_pred, axis=1)    # Convert to class indices
    else:
        train_pred = model.predict(X_train.reshape(X_train.shape[0], -1))  # 1D predictions for scikit-learn models
        test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    
    # Convert y_train and y_test to class indices
    y_train_indices = np.argmax(y_train, axis=1) if y_train.ndim == 2 else y_train
    y_test_indices = np.argmax(y_test, axis=1) if y_test.ndim == 2 else y_test
    
    # Overall accuracy
    train_acc = np.mean(train_pred == y_train_indices)
    test_acc = np.mean(test_pred == y_test_indices)
    
    print(f"Overall Train Accuracy: {train_acc:.4f}")
    print(f"Overall Test Accuracy: {test_acc:.4f}")
    
    # Per-class accuracy
    num_classes = y_train.shape[1] if y_train.ndim == 2 else len(np.unique(y_train))
    print("\nPer-Class Accuracy:")
    for cls in range(num_classes):
        train_cls_indices = y_train_indices == cls
        test_cls_indices = y_test_indices == cls
        train_cls_acc = np.mean(train_pred[train_cls_indices] == cls) if np.any(train_cls_indices) else 0
        test_cls_acc = np.mean(test_pred[test_cls_indices] == cls) if np.any(test_cls_indices) else 0
        print(f"Class {cls}: Train Accuracy = {train_cls_acc:.4f}, Test Accuracy = {test_cls_acc:.4f}")