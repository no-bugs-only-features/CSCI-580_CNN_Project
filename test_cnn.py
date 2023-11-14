# This program tests a CNN model, which should be provided in the command line.

import sys
import tensorflow as tf

# Make sure a model has been provided
if (len(sys.argv) < 2):
    print("Please provide at least one CNN model: python test_cnn.py myCNN.model")
    quit()

# Load and normalize the test data
data = tf.keras.datasets.mnist
(training_pixels, training_labels), (test_pixels, test_labels) = data.load_data()
test_pixels = tf.keras.utils.normalize(test_pixels, axis=1)

# Load and test each provided model
for i in range(1, len(sys.argv)):

    print("\nModel", i)

    try:
        cnn = tf.keras.models.load_model(sys.argv[i])
    except Exception as e:
        print(f"Unable to test {sys.argv[i]}\nError: {e}\n")
    else:
        loss, accuracy = cnn.evaluate(test_pixels, test_labels)
        print("Accuracy:", accuracy)
        print("Loss:", loss, "\n")
