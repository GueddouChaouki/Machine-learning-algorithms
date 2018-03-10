# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 00:55:03 2018

@author: gueddou chaouki
"""
from __future__ import print_function
import matplotlib,sys
from matplotlib import pyplot as plt
import numpy as np
import random



def data_reader(filename):
    to_binary = {"?": 3, "y": 2, "n": 1}
    labels = {"democrat": 1, "republican": -1}

    data = []
    for line in open(filename, "r"):
        line = line.strip()

        label = int(labels[line.split(",")[0]])
        observation = np.array([to_binary[obs] for obs in line.split(",")[1:]] + [1])
        data.append((label, observation))

    return data

data = data_reader(r"C:\Users\pc\Google Drive\Master 1 versailles\2 ème semestre_\Int App\TP 3 Perceptron\house-votes-84.data")
print(data[0:3])
def column(matrix, i):
    return [row[i] for row in matrix]

print(column(data[0:1], 0))
print(column(data[0:1], 1))

random.shuffle(data)

data_Obs = column(data, 1)
trainObs = column(data[0:335], 1)
testObs = column(data[335:435], 1)

len(data_Obs)
len(trainObs)
len(testObs)

print(testObs)

print(column(data[0:1], 0))
print(column(data[0:1], 1))

#Why do we add a constant feature?
# pour eviter de passer par l'origine 00

'''
def classify(dataset, vector):
    activation = 0.0
    for i,w in zip(dataset,vector):
        activation += i*w #np.dot(i, w)
    for i in range(0, len(dataset)):
      activation = sum(xx * yy for xx, yy in zip(dataset[i], vector))
      print(activation)
    return 1.0 if activation >= 0.0 else 0.0
  '''

def classify(observation, vector):
    activation = np.dot(observation, vector)
    return 1 if activation >= 0 else -1

#classify(test[8], [25, -12, 67, -104, -43, 46, -18, -10, 45, -33, 54, -39, 43, -19, 5, -2, 55])


def test(dataset, vector):
  error_rate = 0.0
  for i in range(0, len(dataset)):
    current_obs = column(dataset[i:i+1], 1)
    predicted_value = classify(current_obs,vector)
    real_label = column(dataset[i:i+1], 0)

    if str(real_label) != "["+str(predicted_value)+"]":
      error_rate += 1.0
  return error_rate/float(len(dataset))

vector = [2, -0, 7, -14, 3, 0, -18, 0, 0, +0, 0, 9, -0, -19, 5, -2, 55]
test(data, vector)

#we get better results with this vector yay
vector = [25, -12, 67, -104, -43, 46, -18, -10, 45, -33, 54, -39, 43, -19, 5, -2, 55]
test(data, vector)


def learn(training_set,test_set, weights, Nbr_epoch):
  error_train = []
  error_test = []
  vectors_of_weights = []
  print(weights)
  vectors_of_weights.append(weights)
  error_train.append(test(training_set,weights))
  error_test.append(test(test_set, weights))

  for i in range(0, Nbr_epoch):
    random.shuffle(training_set)
    trainObs = column(training_set, 1)

    for j in range(1, len(training_set)):

      predicted_value = classify(weights,trainObs[j])
      real_label = column(training_set[j:j+1], 0)

      if str(real_label) != "["+str(predicted_value)+"]":
        weights += real_label * trainObs[j]

    vectors_of_weights.append(weights)
    error_train.append(test(training_set,weights))
    error_test.append(test(test_set, weights))

  return weights, vectors_of_weights, error_train, error_test

weights = np.zeros(len(vector))

w2 , _ , p_train, p_test = learn(data[0:335], data[335:435], weights, 12)

print(w2)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.style.use('ggplot')

x = np.arange(1, len(p_train) + 1)
print(x)

fig = Figure()
FigureCanvas(fig)

ax = fig.add_subplot(111)
ax.plot(x, p_train, "rx", label="train")
ax.plot(x, p_test, "bo", label="test")
ax.set_xlabel("Nbr of epoch")
ax.set_ylabel("precesion")
ax.legend()

fig.savefig("learning_curve.pdf")




'''
this part will be about  1.3 Stopping criterion
'''
#iterates as long as there are errors on the training set;

def first_method(training_set, weights):
  print(weights)
  while test(training_set,weights) > 0.05:
    random.shuffle(training_set)
    trainObs = column(training_set, 1)
    for j in range(1, len(training_set)):
      predicted_value = classify(weights,trainObs[j])
      real_label = column(training_set[j:j+1], 0)
      if str(real_label) != "["+str(predicted_value)+"]":
        weights += real_label * trainObs[j]
      print(' weights '+str(weights)+' and error on training set is '+str(test(training_set,weights)))
  return weights, test(training_set,weights)


weights = np.zeros(len(vector))
w1, _ = first_method(data[0:335], weights)

#this method is overfitting the train set :(

#ok the second strategy is implemented on the first part so no need to repeat it


#iterates as long as the error on a validation set decreases (the validation set has to
#be different from the train and test set).


def third_method(training_set,validation_set,test_set, weights):
  previous_error_on_validation_set = 1
  print(weights)
  while test(validation_set,weights) <= previous_error_on_validation_set:
    previous_error_on_validation_set = test(validation_set,weights)
    random.shuffle(training_set)
    print('error on validation set is '+str(previous_error_on_validation_set))
    trainObs = column(training_set, 1)
    for j in range(1, len(training_set)):
      predicted_value = classify(weights,trainObs[j])
      real_label = column(training_set[j:j+1], 0)
      if str(real_label) != "["+str(predicted_value)+"]":
        weights += real_label * trainObs[j]

  print('error on training set is set is '+str(test(training_set,weights)))
  print('error on the test set is set is '+str(test(test_set,weights)))
  return weights

random.shuffle(data)
train = data[0:261]
valid = data[261:361]
test1 = data[361:435]

w3 = third_method(train,valid,test1,weights)


print('error on training set is set is '+str(test(test1,w1)))
print('error on training set is set is '+str(test(test1,w2)))
print('error on training set is set is '+str(test(test1,w3)))

# in the first two methods we were not using a validation set
#the third one seems more efficient because it can give the most approximation error


def spam_reader(filename):
    to_binary = {1: 1, 0: -1}
    data = []
    for line in open(filename, "r"):
        line = line.strip()
        label = to_binary[int(line.split(",")[-1])]
        observation = [float(obs) for obs in line.split(",")[:-1] + [1.0]]

        data.append((label, np.array(observation)))

    return data

spam_data = spam_reader(r"C:\Users\pc\Google Drive\Master 1 versailles\2 ème semestre_\Int App\TP 3 Perceptron\SpamDataSetFolder\spambase.data")

print(spam_data)


def averaged_perceptron_learn(training_set,validation_set, weights):
  print(weights)
  a = np.zeros(len(weights))
  cmpt = 0
  while test(validation_set, weights) > 0.09:
    random.shuffle(training_set)
    trainObs = column(training_set, 1)
    for j in range(1, len(training_set)):
      predicted_value = classify(weights,trainObs[j])
      real_label = column(training_set[j:j+1], 0)
      if str(real_label) != "["+str(predicted_value)+"]":
        weights += real_label * trainObs[j]*0.5
      a = a + weights
      cmpt += 1
    print(test(validation_set, weights))
  return (1.0/float(cmpt)) * a

random.shuffle(spam_data)

train_spam = spam_data[0:2761]
valid_spam = spam_data[2761:3681]
test_spam = spam_data[3681:4601]


sp = column(spam_data[0:1], 1)

print(len(sp[0]))

weights = np.zeros(len(sp[0]))

result_weight = averaged_perceptron_learn(train_spam, valid_spam, weights)

print(test(test_spam,result_weight))
