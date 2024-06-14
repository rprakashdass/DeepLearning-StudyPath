
"""
Original file is located at
    https://colab.research.google.com/drive/1uC6tCMUO0Za1hprsVjiSvP5tC2P_1ZTK
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = data.load_data()
print(x_train.ndim)

x_train.shape

plt.imshow(x_train[0], cmap=plt.cm.binary)
print(y_train[0])

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
model.summary()

print("Model Accuracy:", history.history['accuracy'][-1])

loss =  history.history['loss']
acc =  history.history['accuracy']

plt.plot(loss)
plt.plot(acc)
plt.xlabel('accuracy')
plt.ylabel('loss')
plt.legend(['loss','accuracy',], loc='upper right')

# Predict

predictions = model.predict(x_test)

for i in range(4):
  print("Actual:",y_test[i])
  print("Predicted:",y_test[i])
