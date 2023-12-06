import tensorflow as tf

# Load dataset directly from tensorflow
data = tf.keras.datasets.mnist

# Separate into training data and test data
(training_pixels, training_labels), (test_pixels, test_labels) = data.load_data()
training_pixels = tf.keras.utils.normalize(training_pixels, axis=1)

# Create the CNN
cnn = tf.keras.models.Sequential()

# Add the input layer
cnn.add(tf.keras.layers.Conv2D(1, (14, 14), activation="relu", input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.Conv2D(12, (7, 7), activation="relu"))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compile the CNN
cnn.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

from keras.callbacks import CSVLogger
logger = CSVLogger("K6-Net.csv", append=False)

# Train and save the CNN model
cnn.fit(training_pixels, training_labels, batch_size=64, epochs=40, validation_split=0.03, callbacks=[logger])
cnn.save("K6-Net.model")
