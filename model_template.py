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

# Add the input layer (converts the pixel matrix into an array)
cnn.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Add hidden layers
# cnn.add(tf.keras.layers.LAYER_TYPE(NUM_NODES, activation="ACTIVATION_TYPE"))

# Add the output layer
# Dense layer type means a fully connected layer (can be changed)
# Must be 10 nodes in the layer (one for each possible digit)
# "softmax" activation makes it so the total from all output nodes adds up to 1 
#    (meaning the output value for a digit's node is the probability that
#    the input image depicts that digit)
cnn.add(tf.keras.layers.Dense(10, activation="softmax"))

# Compile the CNN
# optimizer specifies how to update weights based on accuracy (sgd is gradient descent)
# loss calculates how close the cnn's guess is (sparse_categorical_crossentropy
#     is often used for multi-class classification problems like ours)
# metrics should stay consistent across all our models, whatever we choose
cnn.compile(
    optimizer="sgd", 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

# Train and save the CNN model
# epochs is the number of times the model updates itself while training
cnn.fit(training_pixels, training_labels, epochs=5)
cnn.save("template.model")
