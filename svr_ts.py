import os
import pdb
import sys

import numpy
from sklearn import svm
from quant import read_data

def train_svr(dataset=''):
   train, valid, test = read_data(columns=1) 
   x_train = [[x[0] for x in row] for row in train[0]]
   x_test = [[x[0] for x in row] for row in test[0]]


   svr = svm.SVR()
   svr.fit(x_train,train[1])
    
   pred = svr.predict(x_test)
   y = numpy.asarray(test[1], dtype='float32')
   pred = numpy.asarray(pred, dtype='float32')

   cost = ((y-pred)**2).mean()
   print 'Cost on Test sample, size: %d, cost: %f'%(len(x_test),cost)


if __name__ == '__main__':
    train_svr(dataset='table_a.csv')
