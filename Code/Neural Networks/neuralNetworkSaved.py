import numpy as np
from keras.models import Sequential, load_model
from keras.metrics import categorical_accuracy
from keras.layers import Dense, Activation
from sklearn.preprocessing import Imputer
from pandas import read_csv
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mode
# from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import chain
from glob import glob
import os
# cur_dir = os.path.dirname(__file__)
cur_dir = os.getcwd()

def vote(classes,test_data):
    # votes = np.array([np.argmax(t) for c in classes for t in c.predict(test_data)])
    votes = np.array([[np.argmax(t) for t in c.predict(test_data)] for c in classes])
    # votes = transpose(votes)
    winners = np.reshape(mode(votes)[0],-1)
    # print(winners.shape)
    # print(winners)
    # exit()
    return winners


classes = (load_model(file) for file in glob('*.h5'))
# os.chdir('D:\\Models')
# classes = (load_model(file) for file in glob('*.h5'))
# os.chdir(cur_dir)
# classes = chain(classes,(load_model(file) for file in glob('*.h5')))


# a generator would be a bit faster and would save space

# data = read_csv(open('train.csv','r'),na_values='').as_matrix()

# X = data[:,1:-1] # input features
# Y = data[:,-1].astype('int') # input features

# imp = Imputer()#default arguments will suffice
# X = imp.fit(X)


data = read_csv(open('test_knn.csv','r'),na_values='').as_matrix()

X_test = data[:,1:] # input features
# X_test = imp.transform(X_test)
# Y_test = data[:,-1]

predictions = vote(classes,X_test)

with open('pred_ANN_ensemble.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))


# print('Model {0}: f1 = micro:{1}  macro:{2}'.format(i,
# f1_score(Y_test,predictions,average='micro'),
# f1_score(Y_test,predictions,average='macro')))
