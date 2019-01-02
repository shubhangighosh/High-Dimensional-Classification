#header files 
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import cross_validate
import pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#reading data
data = pd.read_csv(open('train.csv','r'),na_values='').as_matrix()
print np.shape(data)
Y = data[:,-1].astype('int') # input labels
Y = np.array(Y)
#summed probabilities for test and train set
df=pd.read_csv('svm_ip.csv', sep=',',header=None)
X = df.values
X = np.array(X)

df=pd.read_csv('svm_test_ip.csv', sep=',',header=None)
X_test = df.values
X_test = np.array(X_test)


#standardisation
X_tot = np.concatenate((X,X_test),axis=0)
X_scaler = StandardScaler()
X_scaler.fit(X_tot)
X = X_scaler.transform(X)
X_test = X_scaler.transform(X_test)

#shuffling data for better cross-validation
X, Y = shuffle(X, Y, random_state=5)

#rbf kernel
svm_clf = SVC(kernel='rbf',C=100,gamma=0.01)
scores = cross_validate(svm_clf, X,Y, cv=5)
print np.mean(scores['test_score'])
print "Done"   
svm_clf.fit(X,Y)
print "Done"   
Y_pred =  svm_clf.predict(X_test)
with open('svm_rbf.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(Y_pred):
        f.write('{0},{1}\n'.format(i,t))
print "Done"        
#linear kernel
svm_clf = svm_clf = SVC(kernel='linear',C=10.0)
scores = cross_validate(svm_clf, X,Y, cv=5)
print np.mean(scores['test_score'])
print "Done"   
svm_clf.fit(X,Y)
print "Done"   
Y_pred =  svm_clf.predict(X_test)
with open('svm_lin.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(Y_pred):
        f.write('{0},{1}\n'.format(i,t))
print "Done"        