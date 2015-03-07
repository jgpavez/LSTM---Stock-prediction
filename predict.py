from collections import OrderedDict
import copy
import cPickle as pkl
import random
import sys
import time
import pdb
import os

import numpy
from pandas.io.parsers import read_csv
from lstm_forex import train_lstm
import datetime

#Change to 2d
res_url = "http://www.google.com/finance/getprices?i=3600&p=2d&f=d,o,h,l,c,v&df=cpct&q="

def read_from_api_hourly(exchange):

    data = read_csv(res_url + exchange, sep=',', header=None, skiprows=7)
    
    data = data.values
    data = data[-10:,1:-1]
    data = numpy.fliplr(data)

    return data
    

# TODO: Continue here
def get_minutes(iterator,date,data,timestamps):
    if data[0][iterator][0] == 'a':
        timestamp = timestamps.pop()
        date = datetime.datetime.fromtimestamp(float(timestamp))
        minutes = date.minute
    else:
        tempdate = date + datetime.timedelta(minutes=float(data[0][iterator]))
        minutes = tempdate.minute
    return minutes,date

def read_from_api_minutes(exchange):
    hours = 11
    res_min = res_url.replace("i=3600","i=60")

    data = read_csv(res_min + exchange, sep=',', header=None, skiprows=7)

    # In case of error because two rows with a
    timestamps = []
    for i in range(0,len(data)):
        if data[0][i][0] == 'a':
            timestamps.append(data[0][i][1:])

    timestamp = timestamps.pop()
    date = datetime.datetime.fromtimestamp(float(timestamp))

    result = numpy.zeros([hours,4])

    iterator = len(data)-1
    minutes,date = get_minutes(iterator,date,data,timestamps)
    tempdate = date + datetime.timedelta(minutes=float(data[0][iterator]))
    print 'Receiving from %s to %s' % (date, tempdate) 

    for i in range(hours-1,-1,-1):
        close = float(data[1][iterator])
        low = close
        high = close
        # TODO: If jump value when minutes is equal to zero??
        while minutes > 0:
            val = float(data[1][iterator])
            high = max(high,val)
            low = min(low,val)
            iterator = iterator - 1
            minutes,date = get_minutes(iterator,date,data,timestamps)

        open = float(data[1][iterator])
        result[i,:] = [open,low,high,close]
        iterator = iterator - 1
        minutes,date = get_minutes(iterator,date,data,timestamps)

    return result


if __name__ == '__main__':

    exchanges = ['AUDJPY', 'CHFJPY', 'EURCHF', 'EURGBP', \
                'EURJPY', 'EURUSD']

    #Data must be normalized by train parameters
    max_len = 10

    model_path = '/user/j/jgpavez/rnn_trading/models/'
    params_path = '/user/j/jgpavez/rnn_trading/data/'
    
    #results = numpy.zeros([len(exchanges),3])
    results = []
    best_ret = None

    for i,exchange in enumerate(exchanges):

        params_file = exchange + '_params.npz' 

        params_file = os.path.join(params_path, params_file)

        params = numpy.load(params_file)
        mean = params['mean']
        std = params['std']

        data = read_from_api_minutes(exchange)
        #only for minutes
        last = data[-1][-1]
        data = data[:-1]

        actual_std = data[:,-1].std()
        data = data - mean
        data = data / std

        x_data = numpy.array([data[:max_len]])

        pred = train_lstm(predict=True, input_pred=x_data, exchange=exchange)    
        un_pred = pred[0][0]*std[0] + mean[0]
        net_return = (un_pred - last)/last

        print 'Results for: %s' % exchange
        print 'Prediction is: %f (normalized), %f ' % (pred[0][0], un_pred)
        print 'Predicted net return : %f , std of data is: %f' % (net_return,actual_std) 
        print data * std + mean
        results.append([exchange, net_return, un_pred, actual_std])
        if best_ret is None or results[best_ret][1] < net_return:
            best_ret = i

    print 'Best results: '
    print results[best_ret]
    print 'All results: '
    for r in results: print r
    #print data*std + mean
