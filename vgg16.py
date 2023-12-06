import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = 	mnist.load_data()

# x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]])/255
# x_test = tf.pad(x_test, [[0,0], [2,2], [2,2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_train = tf.repeat(x_train, 3, axis=3)
x_test = tf.repeat(x_test, 3, axis=3)

from tensorflow.keras import models, layers, losses

model = models.Sequential()

model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))

# Block 1
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Block 2
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Block 3
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Block 4
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

# Block 5
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096, activation="relu"))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

from keras.callbacks import CSVLogger
logger = CSVLogger("LeNet-5.csv")

model.fit(x_train, y_train, batch_size=64, epochs=40, validation_split=0.03, callbacks=[logger])

model.save('VGG-16.model')

# model = models.load_model('LeNet-5.model')
# model.summary()

# model.evaluate(x_test, y_test)