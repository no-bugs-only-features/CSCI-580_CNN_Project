import tensorflow as tf
import numpy as np

# Define the XOR dataset
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
output_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model 
model.fit(input_data, output_data, epochs=10000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(input_data, output_data)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Make the predictions
predictions = model.predict(input_data)
print("\nPredictions:")
for i in range(len(predictions)):
    print(f"Input: {input_data[i]}, Predicted Output: {predictions[i][0]:.4f}")
