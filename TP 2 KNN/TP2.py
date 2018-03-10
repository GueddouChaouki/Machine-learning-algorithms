# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:42:08 2018

@author: pc
"""

import pickle
import gzip
import numpy as np

#load the data set
with open('mnist.pkl','rb') as ifile:
        train, valid, test = pickle.load(ifile, encoding="latin-1")

import matplotlib.pyplot as plt

#print(train[0])
#print(train[1])
im = train[0][0].reshape(28,28)

plt.imshow(im, plt.cm.gray)
plt.show()

print(train[0].shape)

def binarize(im, thres=0.5):
  res =  np.zeros(im.shape)
  res[im >= thres] = 1
  return res

imb = binarize(im)
plt.imshow(imb, plt.cm.gray)
plt.show()
''' on essaye avec T=0.4'''
imb2 = binarize(im, 0.4)
plt.imshow(imb2, plt.cm.gray)
plt.show();
''''''
def compute_horizontal_histogram(im):
  width, height = im.shape
  res = np.zeros(height)
  for i in range (0, height - 1):
    for j in range (0, width -1):
      res[i] = res[i] + im[i][j]
  return res

def compute_vertical_histogram(im):
  width, height = im.shape
  res = np.zeros(width)
  for i in range (0, width - 1):
    for j in range (0, height -1):
      res[i] = res[i] + im[i][j]
  return res

horizontal_histgram = compute_horizontal_histogram(imb)
vertical_histgram = compute_vertical_histogram(imb)
width, height = im.shape
print(horizontal_histgram)
print(vertical_histgram)

img = binarize(train[0][0].reshape(28,28), 0.5)
plt.imshow(img, plt.cm.gray)
plt.show();
print(compute_horizontal_histogram(img))

img = binarize(train[0][4].reshape(28,28), 0.5)
plt.imshow(img, plt.cm.gray)
plt.show();
print(compute_horizontal_histogram(img))

def compute_mean_image(n, m):
  img = binarize(train[0][0].reshape(28,28), 0.5)
  res =  np.zeros(img.shape)
  width, height = img.shape
  print(img.shape)
  for i in range(n, m):
    img = binarize(train[0][i].reshape(28,28), 0.5)

    for j in range(0, width):
      for k in range(0, height):
        res[j][k] += img[j][k]

  for j in range(0, width):
    for k in range(0, height):
      res[j][k] = res[j][k]/(m-n)
  print(res)
  return binarize(res, 0.5)

img = compute_mean_image(0,8)
plt.imshow(img, plt.cm.gray)
plt.show()

