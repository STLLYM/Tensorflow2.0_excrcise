# -*- coding: gbk -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1].replace([2, 3, 4, 5], 0)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())


# 决策树
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X_train, y_train)
# clf.predict_proba(X_test)

#svm
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)
print(clf.support_vectors_)
