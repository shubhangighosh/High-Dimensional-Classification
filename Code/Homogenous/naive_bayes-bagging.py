import numpy as np
import pandas as pd
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier


#df=pd.read_csv('train.csv', sep=',',header=None, low_memory= False)
data = pd.read_csv(open('train.csv','r'),na_values='').as_matrix()
print np.shape(data)
X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features

data_t = pd.read_csv(open('test.csv','r'),na_values='').as_matrix()
print np.shape(data_t)
X_test = data_t[:,1:] # input features



imp = Imputer(missing_values='NaN')#default arguments will suffice
X = imp.fit_transform(X)
X_test = imp.transform(X_test)
#sX = KNN(k=3).complete(X)



X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
X_test = X_scaler.transform(X_test)

#datapoints are very close together --- variance very small --- have to project it to some higher dimension
#also look at other parameters of random forest
#missing values using gbdt
#will projecting to higher dimension help, already no of datapoints less compd to no of dimensions
print np.shape(X)

"""rbf_feature = RBFSampler(gamma=1, random_state=1)
X_train = rbf_feature.fit_transform(X_train)
X_test = rbf_feature.transform(X_test)

print np.shape(X_train)

poly = PolynomialFeatures(2)
X_train = poly.fit_transform(X_train)
X_test = poly.transform(X_test)

avj = X_train.mean(axis=0)  #-------find for each class tomorrow
transformer = FunctionTransformer(np.log1p)"""
print np.shape(X_test)

#use cv on these two params
clf = BaggingClassifier(RandomForestClassifier(n_jobs=-1),max_samples=0.5, max_features=0.5,n_jobs=-1)

clf.fit(X, Y)
#Y_pred =  clf.predict(X_test)
#Y_pred_train =  clf.predict(X)
Y_pred = []
Y_pred_proba =  clf.predict(X_test)
Y_pred.append(Y_pred_proba)
Y_pred = np.reshape(np.array(Y_pred),(2900))
print(Y_pred.shape)
my_df = pd.DataFrame(Y_pred)
my_df.to_csv('shu_rf_pred.csv', index=False, header=False)






