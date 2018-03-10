# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 22:25:45 2018

@author: pc
"""

from sklearn.tree import DecisionTreeClassifier
from data import *
X, Y, dictionary = loadTextDataBinary('data/sentiment.tr')
X.shape
Y.shape
