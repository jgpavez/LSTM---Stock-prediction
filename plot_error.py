import os
import pdb
import sys

import numpy

from numpy import genfromtxt
import matplotlib.pyplot as plt


def getfile(filename, dir='/user/j/jgpavez/lstmTimeSerie/logs/'):
    path = os.path.join(dir,filename)
    return path

def plot_error():
    file1 = getfile('square_128.txt')
    data1 = genfromtxt(file1)

    file2 = getfile('square_64.txt')
    data2 = genfromtxt(file2)

    file3 = getfile('square_32_no_sum.txt')
    data3 = genfromtxt(file3)

    file4 = getfile('square_12.txt')
    data4 = genfromtxt(file4)

    file5 = getfile('square_01_sigm.txt')
    data5 = genfromtxt(file5)

    file6 = getfile('square_32_drop.txt')
    data6 = genfromtxt(file6)

    file7 = getfile('square_001.txt')
    data7 = genfromtxt(file7)

    file8 = getfile('square_32_10i.txt')
    data8 = genfromtxt(file8)

    file9 = getfile('square_32_7i.txt')
    data9 = genfromtxt(file9)

    file10 = getfile('square_32_15i.txt')
    data10 = genfromtxt(file10)

    file11 = getfile('square_32_20i.txt')
    data11 = genfromtxt(file11)


    plt.plot(range(len(data1[0:-1:10])), data1[0:-1:10], 
                    range(len(data2[0:-1:10])), data2[0:-1:10]  
                    ,range(len(data3[0:-1:10])), data3[0:-1:10] 
                    ,range(len(data4[0:-1:10])), data4[0:-1:10]
                    ,range(len(data5[0:-1:10])), data5[0:-1:10]
                    ,range(len(data6[0:-1:10])), data6[0:-1:10]
                    ,range(len(data7[0:-1:10])), data7[0:-1:10]
                    ,range(len(data8[0:-1:10])), data8[0:-1:10]
                    ,range(len(data9[0:-1:10])), data9[0:-1:10]
                    ,range(len(data10[0:-1:10])), data10[0:-1:10]
                    ,range(len(data11[0:-1:10])), data11[0:-1:10]
                    )  

    plt.legend(['LSTM_128_01_30', 'LSTM_64_01_30','LSTM_32_01_30','LSTM_12_01_30',
    'LSTM_32_01_10_SIGM', 'LSTM_32_001_10_DROP', 'LSTM_32_001_10', 'LSTM_32_01_10',
    'LSTM_32_01_7','LSTM_32_01_15','LSTM_32_01_20'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('LSTM MSE')

    plt.savefig('lstm_mse.png')


plot_error()

