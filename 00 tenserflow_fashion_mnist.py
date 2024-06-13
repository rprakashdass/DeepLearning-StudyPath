"""
Original file is located at
    https://colab.research.google.com/gist/rprakashdass/e05c9fefc5fc47c9f39732a86ac21b08/tenserflow_fashion_mnist.ipynb
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

df = keras.datasets.fashion_mnist

(x_train, y_train),( x_test, y_test) = df.load_data()

x_train = x_train /255.0
x_test = x_test/255.0

print(x_train[0])
plt.imshow(x_train[0], cmap=plt.cm.binary)

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',	'Coat', 'Sandal',	'Shirt', 'Sneaker', 'Bag',	'Ankle boot']

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=25)
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss, test_acc)

prediction = model.predict(x_test)
for i in range(10):
  print(prediction[i])
  print("Actual:",labels[y_test[i]])
  print("Predicted:",labels[np.argmax(prediction[i])])
