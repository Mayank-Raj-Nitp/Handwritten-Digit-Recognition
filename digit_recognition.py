import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# ---  DATA LOADING AND PREPARATION ---

# Loading the  MNIST dataset (60,000 training, 10,000 test images)
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
print(f"Loaded MNIST dataset. Train samples: {len(X_train_raw)}")

X_train_norm = X_train_raw.astype('float32') / 255.0
X_test_norm = X_test_raw.astype('float32') / 255.0

# One-Hot Encode the target variable (0-9 digits)
y_train = to_categorical(y_train_raw)
y_test = to_categorical(y_test_raw)
num_classes = y_train.shape[1] # Should be 10

# ---  ANN MODEL DEVELOPMENT AND TRAINING ---

# Building the Sequential Model
model = Sequential([
    # Input Layer: Flattens the 28x28 image into a 784-element vector
    Flatten(input_shape=(28, 28)),
    # First Hidden Layer
    Dense(units=128, activation='relu'),
    # Second Hidden Layer
    Dense(units=64, activation='relu'),
    # Output Layer (Softmax for multi-class classification)
    Dense(units=num_classes, activation='softmax')
])

# Compile the model
# Optimizer: adam (standard), Loss: categorical_crossentropy (for one-hot targets)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (Using small epochs to meet time constraint)
print("\n--- Training the MNIST ANN (Image Data) ---")
model.fit(
    X_train_norm, y_train,
    epochs=5,           # Small number of epochs for quick execution
    batch_size=128,
    validation_split=0.1,
    verbose=0
)

# ---  EVALUATION ---
loss, accuracy = model.evaluate(X_test_norm, y_test, verbose=0)

print("--- Model Training Complete ---")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.4f}")
model.summary(print_fn=lambda x: print("ANN Model Summary:\n" + x.replace('\n', '\n  ')))

# Predict the first image in the test set
prediction_probabilities = model.predict(X_test_norm[np.newaxis, 0], verbose=0)
predicted_class = np.argmax(prediction_probabilities)
true_class = y_test_raw[0]

print(f"\nFirst Test Image Prediction:")
print(f"  True Digit: {true_class}")
print(f"  Predicted Digit: {predicted_class}")
print(f"  Confidence: {prediction_probabilities[0][predicted_class]*100:.2f}%")
