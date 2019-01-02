from fancyimpute import SoftImpute
from pandas import read_csv
import pandas as pd
import numpy as np

if True:
    data = read_csv(open('train.csv','r'),na_values='').as_matrix()
    X1 = data[:,1:-1] # input features
    Y1 = data[:,-1].astype('int') # input features


    # print(X1.shape,np.transpose(Y1).shape)
    X1 = SoftImpute().complete(X1)


    # print(X1.shape,np.transpose(Y1).shape)

    # pd.DataFrame(X1).to_csv('X1.csv', header = None)

    # X1 = read_csv(open('X1.csv','r'),na_values='').as_matrix()
    # X1 = np.loadtxt('X1.csv')

    # print(X1.shape,Y1[np.newaxis,:].shape),np.reshape(np.arange(9501),(-1,1)).shape
    # train = np.concatenate([np.arange(9501)[np.newaxis,:],X1,Y1[np.newaxis,:]],axis=1)
    # train = np.concatenate([np.arange(9501).T,X1,Y1.T],axis=1)
    train = np.concatenate((np.reshape(np.arange(9501),(-1,1)),X1,np.reshape(Y1,(-1,1))),axis=1)
    pd.DataFrame(train).to_csv('train_SI.csv', header = None)


    print('Train done:',train.shape,data.shape)


    data = read_csv(open('test.csv','r'),na_values='').as_matrix()
    X2 = data[:,1:] # features

    train = X1.shape[0]

    X = np.concatenate((X1,X2))
    # print(X.shape,X1.shape,X2.shape)
    del X1,X2

    X_net = SoftImpute().complete(X)
    del X

    # pd.DataFrame(X_net).to_csv('all.csv', header = None)

    test = X_net[train:]
    del X_net
    test = np.concatenate((np.reshape(np.arange(2900),(-1,1)),test),axis=1)

    # print(train.shape,test.shape)

    pd.DataFrame(test).to_csv('test_SI.csv', header = None)

    print('Test done:',test.shape,data.shape)


# data = read_csv(open('test.csv','r'),na_values='').as_matrix()
# test = np.concatenate((np.reshape(np.arange(2900),(-1,1)),test),axis=1)


