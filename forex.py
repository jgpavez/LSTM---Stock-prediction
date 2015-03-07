import os 
import pdb
import sys

import numpy

import theano
from numpy import genfromtxt
from pandas import Series

#import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from sklearn import preprocessing

def data_preprocessing(data):
    #data = data[51000:]
    #data = data[350:,:]
    # Standarization
    
    # Compute compound return serie
    #data = numpy.log(data / numpy.roll(data, 1, axis = 0))
    #data = data[1:,:]
    #data = data / numpy.roll(data,1,axis=0)
    #data = data[1:,:] - 1.
    
    print data.shape

    #print
    #ts = Series(numpy.ravel(data[:,0]))
    #ts.plot()
    #plt.show()


    mean = data.mean(axis=0)
    std = data.std(axis=0)

    data = data - mean
    data = data/std

    #Some kind of smoothing??

    #min_max = preprocessing.MinMaxScaler()
    #data = min_max.fit_transform(data)

    #Put between 1 and 0
    return data,mean,std

def read_data(path="AUDJPY_hour.csv", dir="/user/j/jgpavez/rnn_trading/data/",
        max_len=30, valid_portion=0.1, columns=4, up=False, params_file='params.npz' ):

    path = os.path.join(dir, path)

    data = genfromtxt(path, delimiter=',',skip_header=1)

    data = data[:,3:(3+columns)]

    data,mean,std = data_preprocessing(data)

    # Save data parameters
    numpy.savez(params_file, mean=mean, std=std)

    x_data = numpy.array([data[i:i+max_len,:] for i in xrange(len(data)-max_len)])
    y_data = numpy.array([data[i][3] for i in xrange(max_len , len(data))])

    if up is True:
        y_data = y_data > x_data[:,-1,0]
        y_data = numpy.asarray(y_data, dtype='int64')

    # split data into training and test
    train_set_x, test_set_x, train_set_y, test_set_y = cv.train_test_split(x_data, 
    y_data, test_size=0.2, random_state=0)

    # split training set into validation set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test, mean, std 

def prepare_data(seqs, labels, steps, x_dim, up=False):
    n_samples = len(seqs)
    max_len = steps
    x = numpy.zeros((max_len, n_samples, x_dim)).astype('float32')
    if up is True:
        y = numpy.asarray(labels, dtype='int64')
    else:
        y = numpy.asarray(labels, dtype='float32')

    for idx, s in enumerate(seqs):
        x[:,idx,:] = s

    return x, y

