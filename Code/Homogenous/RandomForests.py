from sklearn.model_selection import StratifiedKFold
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pandas import read_csv

data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()

X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features




for train_i,test_i in StratifiedKFold(n_splits=3,random_state=3).split(X,Y):
    cl = ExtraTreesClassifier(n_estimators=100, n_jobs = -1)
    np.random.shuffle(train_i)
    X_train = X[train_i]
    Y_train = Y[train_i]
    cl.fit(X_train,Y_train)

    X_test = X[test_i]
    Y_test = Y[test_i]

    print('Score:',cl.score(X_test,Y_test))

