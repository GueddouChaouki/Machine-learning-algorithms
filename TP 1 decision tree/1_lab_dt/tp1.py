# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:25:45 2018

@author:
"""

from sklearn.tree import DecisionTreeClassifier
from data import *
X, Y, dictionary = loadTextDataBinary('data/sentiment.tr')
X.shape
X[0][1:]
Y.shape
Y[:10]
len(dictionary)
dictionary[:10]

dt = DecisionTreeClassifier(max_depth = 1)
dt.fit(X,Y)
showTree(dt, dictionary)

#Depth 1 accuracy
np.mean(dt.predict(X)==Y)

dt = DecisionTreeClassifier(max_depth = 2)
dt.fit(X,Y)
showTree(dt, dictionary)

#Depth 2 accuracy
np.mean(dt.predict(X)==Y)

dt = DecisionTreeClassifier(max_depth = 3)
dt.fit(X,Y)
showTree(dt, dictionary)

#Depth 3
np.mean(dt.predict(X)==Y)

Xtest, Ytest, _ = loadTextDataBinary('data/sentiment.te',dictionary)
np.mean(dt.predict(Xtest) == Ytest)

X_test, Y_test, _ = loadTextDataBinary('data/sentiment.de',dictionary)
np.mean(dt.predict(X_test) == Y_test)

#training error
for i in range(1, 21):
  dt = DecisionTreeClassifier(max_depth = i)
  dt.fit(X,Y)
  print("epoch number: "+ str(i))
  print("train set         : "+str(np.mean(dt.predict(X)==Y)))
  print("test set          : "+str(np.mean(dt.predict(Xtest) == Ytest)))
  print("develeppemenet set: "+str(np.mean(dt.predict(X_test) == Y_test)))

""" the plot part """
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.style.use('ggplot')

X_train, Y_train, dictionary = loadTextDataBinary('data/sentiment.tr')
X_test, Y_test, _ = loadTextDataBinary('data/sentiment.te',dictionary)
X_dev, Y_dev, _ = loadTextDataBinary('data/sentiment.de',dictionary)

def evaluate(depth):
  clf = DecisionTreeClassifier(max_depth = depth)
  clf.fit(X_train , Y_train)
  return np.mean(clf.predict(X_train) == Y_train), \
         np.mean(clf.predict(X_dev) == Y_dev), \
         np.mean(clf.predict(X_test) == Y_test)

precisions = [evaluate(d) for d in range(1,21)]

p_train,p_dev,p_test = zip(*precisions)

x = np.arange(1, len(p_train) + 1)
print(x)

fig = Figure()
FigureCanvas(fig)

ax = fig.add_subplot(111)
ax.plot(x, p_train, "r+", label="train")
ax.plot(x, p_dev, "gx", label="dev")
ax.plot(x, p_test, "bo", label="test")
ax.set_xlabel("depth")
ax.set_ylabel("precesion")
ax.legend()

fig.savefig("learning_curve.pdf")
