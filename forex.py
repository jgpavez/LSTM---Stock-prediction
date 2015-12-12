'''
 Code to read from forex data
 data can be obtained from http://www.histdata.com/download-free-forex-data/
 both daily data and minute data can be used
'''

# Minute data is under implementation


import os 
import pdb
import sys

import numpy

import theano
from numpy import genfromtxt
from pandas import Series
import datetime
import csv

#import matplotlib.pyplot as plt
import sklearn.cross_validation as cv
from sklearn import preprocessing


def data_preprocessing(data):
    '''
    Simple preprocessing of data
    '''
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
        max_len=30, valid_portion=0.1, columns=4, up=False, params_file='params.npz',min=False):

    '''
    Reading forex data, daily or minute
    '''
    path = os.path.join(dir, path)
    
    #data = read_csv(path,delimiter=delimiter)
    data = genfromtxt(path, delimiter=',',skip_header=1)
    # Adding data bu minute
    if min == False:
        date_index = 1
        values_index = 3
        hours = data[:,2]
    else:
        date_index = 0
        values_index = 1
    
    dates = data[:,date_index]
    days = numpy.array([datetime.datetime(int(str(date)[0:-2][0:4]),int(str(date)[0:-2][4:6]),
                    int(str(date)[0:-2][6:8])).weekday() for date in dates])
    months = numpy.array([datetime.datetime(int(str(date)[0:-2][0:4]),int(str(date)[0:-2][4:6]),
                    int(str(date)[0:-2][6:8])).month for date in dates])

    #dates[:,date_index] = days

    data = data[:,values_index:(values_index+columns)]

    data,mean,std = data_preprocessing(data)

    # Save data parameters
    numpy.savez(params_file, mean=mean, std=std)

    #x_data = numpy.array([data[i:i+max_len,:] for i in xrange(len(data)-max_len)])
    #y_data = numpy.array([data[i][-1] for i in xrange(max_len , len(data))])

    # Not consider jumps between days of market closing
    #TODO: Here I'm just considering weekends, have to think about holydays
    x_data = []
    y_data = []
    for i in xrange(len(data)-max_len):
        #TODO: just working for max_len < 24
        if (dates[i+max_len-1] == 4 and dates[i+max_len] <> 4):
            continue
        x_data.append(data[i:i+max_len,:]) 
        y_data.append(data[i+max_len][-1])
    x_data = numpy.array(x_data)
    y_data = numpy.array(y_data)

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

