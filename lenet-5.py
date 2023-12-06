import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

from tensorflow.keras import layers, models, losses

model = models.Sequential()
model.add(layers.experimental.preprocessing.Resizing(32, 32, interpolation="bilinear", input_shape=x_train.shape[1:]))
model.add(layers.Conv2D(6, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

from keras.callbacks import CSVLogger
logger = CSVLogger("LeNet-5.csv")

model.fit(x_train, y_train, batch_size=64, epochs=40, validation_split=0.03, callbacks=[logger])

model.save('LeNet-5.model')

# model = models.load_model('LeNet-5.model')
# model.summary()

model.evaluate(x_test, y_test)