import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = 	mnist.load_data()

x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]])/255
x_test = tf.pad(x_test, [[0,0], [2,2], [2,2]])/255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)

print(x_test.shape)

x_train = tf.repeat(x_train, 3, axis=3)
x_test = tf.repeat(x_test, 3, axis=3)

print(x_train.shape)

x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[:-2000,:,:,:]
y_train = y_train[:-2000]

from tensorflow.keras import models, layers, losses

model = models.Sequential()

model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=x_train.shape[1:]))

model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))

model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
model.add(layers.Lambda(tf.nn.local_response_normalization))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(3, strides=2))

model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
model.add(layers.Activation('relu'))

model.add(layers.Flatten())

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))

model.save("AlexNet.model")

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

# model = models.load_model("AlexNet.model")

# model.evaluate(x_test, y_test)