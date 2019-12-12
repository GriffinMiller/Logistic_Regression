# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:18:13 2019

@author: Griffin
"""
import numpy as np
import math

def load_data(dataloc):
    with open(dataloc) as f:
        lines = f.read().splitlines()
        arr  = []
        for line in lines:
            if(len(line) != 0):
                arr.append(line.split(','))
        return arr

def rearange_data(data):
    # replaces player name with biaary classification at first index
    # and then removes the binary classification at the last index 
    arr = []
    n = len(data[0]) - 1
    for line in data:
        line[0] = '-1' if float(line[n]) < 1 else '1'
        line.pop(n)
        line.insert(1,1)
        arr.append(line)
    return arr

def data_to_float(data):
    arr = []
    for line in data:
        try:
            a = np.asarray(line).astype(np.float)
            if(len(arr) == 0): arr = a
            else: arr = np.vstack((arr,a))
        except Exception:
            pass
    return arr


def get_manipulated_data(dataloc):
    data = data_to_float( rearange_data( load_data(dataloc) ) )
    return data

def get_train_and_test_data(data):
    percent_train = .7
    n = len(data)
    train_index_range = int( n*percent_train )
    train  = data[0:train_index_range-1]
    test = data[train_index_range:n-1]
    return train,test

def split_data(data_set):
    return np.array(data_set)[:,1:], np.array(data_set)[:,0]


#traindataloc = "../data/test.txt"
#data = get_manipulated_data(traindataloc)
#train,test = get_train_and_test_data(data)
#print(len(train))
#print(len(test))
#print(test)