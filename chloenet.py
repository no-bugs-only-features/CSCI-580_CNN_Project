# This is a template for everyone to use when making CNN models. It's
# based off of the tutorial video included in our citations, and extra comments
# have been included in an attempt to thoroughly explain what each part does.
# When using the template to create a new CNN model, you'll need to add some
# number of hidden layers; you can also change the output layer type and the
# model's optimizer and loss functions.

import tensorflow as tf

# Load dataset directly from tensorflow
# Dataset is composed of 28px by 28px labeled images
data = tf.keras.datasets.mnist

# Separate into training data and test data
(training_pixels, training_labels), (test_pixels, test_labels) = data.load_data()

# Normalize pixel information (convert pixel rgb values from a number between 
# 0-255 to a number between 0-1, indicating how dark/light the pixel is)
training_pixels = tf.keras.utils.normalize(training_pixels, axis=1)

# Create the CNN
cnn = tf.keras.models.Sequential()

# Add the input layer
# "Conv2D" means this is a convolutional layer
# 10 is the number of filters, (4,4) is the size of each filter
# Change layer type, activation, and filter number/size as you like

#
"""cnn.add(tf.keras.layers.Conv2D(10, (4, 4), activation="sigmoid", input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(64, (2, 2), activation='sigmoid'))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid'))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='tanh'))"""

# Add hidden layers
#95.76% accuracy
"""
cnn.add(tf.keras.layers.Conv2D(10, (4, 4), activation="relu", input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(64, (2, 2), activation='sigmoid'))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='sigmoid'))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='tanh'))"""

## 98.08% after 5 epochs, 98.85 after 10 epochs
cnn.add(tf.keras.layers.Conv2D(10, (4, 4), activation="relu", input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(64, (2, 2), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# Add the output layer
# Dense layer type means a fully connected layer (this is optional, but if
#    a dense layer is used, there must be a layer to flatten the cnn -- which
#    turns the matrix into an array)
# Must be 10 nodes in the layer (one for each possible digit)
# "softmax" activation makes it so the total from all output nodes adds up to 1 
#    (meaning the output value for a digit's node is the probability that
#    the input image depicts that digit)
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compile the CNN
# optimizer specifies how to update weights based on accuracy (sgd is gradient descent)
# loss calculates how close the cnn's guess is (sparse_categorical_crossentropy
#     is often used for multi-class classification problems like ours)
# metrics should stay consistent across all our models, whatever we choose
cnn.compile(
    optimizer="adam", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

from keras.callbacks import CSVLogger
logger = CSVLogger("ChloeNet.csv", append=False)

# Train and save the CNN model
# epochs is the number of times the model updates itself while training
cnn.fit(training_pixels, training_labels, batch_size=64, epochs=40, validation_split=0.03, callbacks=[logger])
cnn.save("ChloeNet.model")
