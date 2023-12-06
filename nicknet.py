import tensorflow as tf

# Load data from tensorflow
data = tf.keras.datasets.mnist
(training_pixels, training_labels), (test_pixels, test_labels) = data.load_data()

# Normalize pixel information
training_pixels = tf.keras.utils.normalize(training_pixels, axis=1)
test_pixels = tf.keras.utils.normalize(test_pixels, axis=1)

# Create CNN
cnn = tf.keras.models.Sequential()

# Add the input layer
cnn.add(tf.keras.layers.Conv2D(32, (4, 4), activation="relu", input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
# add all hidden layers 
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation="relu"))
cnn.add(tf.keras.layers.Dropout(0.5))
cnn.add(tf.keras.layers.Dense(64, activation="relu"))

# Output layer
cnn.add(tf.keras.layers.Dense(10, activation="softmax"))

cnn.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

from keras.callbacks import CSVLogger
logger = CSVLogger("NickNet.csv", append=False)

cnn.fit(training_pixels.reshape(-1, 28, 28, 1), training_labels, batch_size=64, epochs=40, validation_split=0.03, callbacks=[logger])

# Evaluate the model on the a test set
# test_loss, test_acc = cnn.evaluate(test_pixels.reshape(-1, 28, 28, 1), test_labels)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
cnn.save("NickNet.model")
