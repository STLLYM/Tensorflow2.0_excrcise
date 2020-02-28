import tensorflow as tf
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split



data = pd.read_csv('data.csv')
#print(data.head())
count_y = data.iloc[:, -1].value_counts()
# print(count_y)

x = data.iloc[:, 1:-1]
y = data.iloc[:, -1].replace([2, 3, 4, 5], 0)
# print(y.value_counts())
x = x.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



modle = tf.keras.Sequential()
modle.add(tf.keras.layers.Dense(128, input_shape=(178,), activation='relu'))
# modle.add(tf.keras.layers.Dropout(0.5))
modle.add(tf.keras.layers.Dense(128, activation='relu'))
# modle.add(tf.keras.layers.Dropout(0.5))
modle.add(tf.keras.layers.Dense(128, activation='relu'))
# modle.add(tf.keras.layers.Dropout(0.5))
modle.add(tf.keras.layers.Dense(128, activation='relu'))
# modle.add(tf.keras.layers.Dropout(0.5))
modle.add(tf.keras.layers.Dense(128, activation='relu'))
# modle.add(tf.keras.layers.Dropout(0.5))
modle.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# print(modle.summary())

modle.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
)
modle.fit(X_train, y_train, epochs=30)
print("test")
modle.evaluate(X_test, y_test)
