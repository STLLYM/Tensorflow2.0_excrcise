# -*- coding: gbk -*-

import tensorflow as tf
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
import joblib as jl

data = pd.read_csv('data.csv')
#print(data.head())
count_y = data.iloc[:, -1].value_counts()
# print(count_y)

x = data.iloc[:, 1:-1]
y = data.iloc[:, -1].replace([2, 3, 4, 5], 0)
x = x.values
y = y.values
print(type(x))
print(type(y))
# print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 直接全连接
# 搭建两层大小为128的全连接层，输出采样sigmoid，可解释为概率大小
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(178,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

jl.dump(model, 'modle1.pkl')
print("enter sucess")

clf2 = jl.load('modle1.pkl')
clf2.fit(X_train, y_train)
print(clf2.support_vectors_)