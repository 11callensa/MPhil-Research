import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split


def neural_network():
    # Generate synthetic data
    X = np.random.rand(1000, 10)
    y = np.random.rand(1000, 1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test MAE: {mae}")

    # Make predictions
    predictions = model.predict(X_test[:5])
    print(predictions)

    return predictions
