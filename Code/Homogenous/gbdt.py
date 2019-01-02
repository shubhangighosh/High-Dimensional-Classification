#Header filess
import numpy as np
import pandas as pd
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_validate

#Reading data
data = pd.read_csv(open('train.csv','r'),na_values='').as_matrix()
print np.shape(data)
X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features

data_t = pd.read_csv(open('test.csv','r'),na_values='').as_matrix()
print np.shape(data_t)
X_test = data_t[:,1:] # input features


#Imputing data
imp = Imputer(missing_values='NaN')#default arguments will suffice
X = imp.fit_transform(X)
X_test = imp.transform(X_test)


#standardising data
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
X_test = X_scaler.transform(X_test)


#use classifier

clf =  GradientBoostingClassifier(n_estimators=10,min_samples_split=70,min_samples_leaf=50)

clf.fit(X, Y)
#predictions
predictions = clf.predict(X_test)
with open('rf.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))





