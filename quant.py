import os 
import pdb
import sys

import numpy
from matplotlib import pyplot

import theano
from numpy import genfromtxt
from pandas import Series

import sklearn.cross_validation as cv

def data_preprocessing(data):
    data = data[350:,:]
    # Standarization
    data = data - data.mean(axis=0)
    data = data/data.std(axis=0)
    #Some kind of smoothing??

    return data 

def read_data(path="table_a.csv", dir="/user/j/jgpavez/lstmTimeSerie/data/",
        max_len=30, valid_portion=0.1 ):
    path = os.path.join(dir, path)

    data = genfromtxt(path, delimiter=',')
    data = data[:,2:-1]

    data = data_preprocessing(data)

    x_data = numpy.array([data[i:i+max_len,:] for i in xrange(len(data)-max_len)])
    y_data = numpy.array([data[i][0] for i in xrange(max_len , len(data))])
    
    # split data into training and test
    train_set_x, test_set_x, train_set_y, test_set_y = cv.train_test_split(x_data, 
    y_data, test_size=0.3, random_state=0)

    # split training set into validation set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test 

def prepare_data(seqs, labels, steps, x_dim):
    n_samples = len(seqs)
    max_len = steps
    x = numpy.zeros((max_len, n_samples, x_dim)).astype('float32')
    y = numpy.asarray(labels, dtype='float32')

    for idx, s in enumerate(seqs):
        x[:,idx,:] = s

    return x, y

