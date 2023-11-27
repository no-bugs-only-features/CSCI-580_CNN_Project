import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0,0], [2,2], [2,2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

x_val = x_train[-2000:,:,:,:] 
y_val = y_train[-2000:] 
x_train = x_train[:-2000,:,:,:] 
y_train = y_train[:-2000]

from tensorflow.keras import layers, models, losses

# model = models.Sequential()
# model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
# model.add(layers.AveragePooling2D(2))
# model.add(layers.Activation('sigmoid'))
# model.add(layers.Conv2D(16, 5, activation='tanh'))
# model.add(layers.AveragePooling2D(2))
# model.add(layers.Activation('sigmoid'))
# model.add(layers.Conv2D(120, 5, activation='tanh'))
# model.add(layers.Flatten())
# model.add(layers.Dense(84, activation='tanh'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# model.save('LeNet-5.model')

model = models.load_model('LeNet-5.model')

model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))

fig, axis = plt.subplots(2, 1, figsize=(15,15))
axis[0].plot(history.history['loss'])
axis[0].plot(history.history['val_loss'])
axis[0].title.set_text('Training Loss vs Validation Loss')
axis[0].legend(['Train', 'Val'])

axis[1].plot(history.history['accuracy'])
axis[1].plot(history.history['val_accuracy'])
axis[1].title.set_text('Training Loss vs Validation Accuracy')
axis[1].legend(['Train', 'Val'])

plt.show()

#model.evaluate(x_test, y_test)