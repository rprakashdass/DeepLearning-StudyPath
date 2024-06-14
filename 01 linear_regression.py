"""
Original file is located at
    https://colab.research.google.com/drive/1oLilpJqaqZ6DqzMfepwdwqXswxtZygKw
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("housing.csv", delim_whitespace=True)
labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

print(df.shape)
print(df.ndim)

df.columns = labels
df = df.rename(columns={'MEDV' : 'PRICE'})
df.describe()

x = df.drop(columns='PRICE' ,axis=1)
y = df.PRICE

df['INDUS'].plot(kind='hist', bins=20, title='INDUS')
plt.gca().spines[['top', 'right',]].set_visible(False)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)
# print(x_test)

scalar = StandardScaler()
scalar.fit(x_train)

x_train_scaled = scalar.transform(x_train)
x_test_scaled = scalar.transform(x_test)

model = keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear'),
])

model.compile(optimizer='adam',loss='mean_squared_error', metrics=['mae'])
history = model.fit(x_train_scaled, y_train, validation_split=0.2 ,epochs=150)

model.summary()

model.evaluate(x_test_scaled, y_test)

print("Model Accuracy:", history.history['mae'][-1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
