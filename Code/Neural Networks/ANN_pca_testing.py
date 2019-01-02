import numpy as np
from keras.models import Sequential, load_model
from keras.metrics import categorical_accuracy
from keras.layers import Dense, Activation
from sklearn.preprocessing import Imputer
from pandas import read_csv
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mode
from sklearn.metrics import f1_score
from keras.regularizers import l1,l2
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.constraints import maxnorm
from sklearn.model_selection import StratifiedShuffleSplit
import os
cur_dir = os.path.dirname(__file__)


def vote(classes,test_data):
    votes = np.array([[np.argmax(t) for t in c.predict(test_data)] for c in classes])
    winners = np.reshape(mode(votes)[0],-1)
    return winners

data = read_csv(open('train_pca.csv','r'),na_values='').as_matrix()

X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features
Y1 = to_categorical(Y)

classes = []

imp = Imputer()#default arguments will suffice
X = imp.fit_transform(X)

# Dropout(rate, noise_shape=None, seed=None)

i = 0
for train_i,test_i in StratifiedShuffleSplit(n_splits=3,random_state=None).split(X,Y):
    np.random.shuffle(train_i)
    X_train = X[train_i]
    Y_train = Y1[train_i]
    brain = Sequential()
    brain.add(Dense(871, input_dim = 871, activation = 'relu', kernel_regularizer=l2(1e-2)))
    brain.add(Dropout(0.1))
    brain.add(Dense(497, activation = 'relu', kernel_regularizer=l2(1e-1)))
    # brain.add(Dense(1197, activation = 'relu', kernel_constraint=maxnorm(5), kernel_regularizer=l2(0.01)))
    brain.add(Dropout(0.1))
    brain.add(Dense(29, activation = 'softmax'))
    opt = Adam(lr = 0.1)
    brain.compile(loss='categorical_crossentropy', optimizer = 'adam',
     metrics=[categorical_accuracy])

    brain.fit(X_train,Y_train,epochs=50, batch_size=30, verbose=2, validation_split = 0.2)

    X_test = X[test_i]
    Y_test = Y[test_i]

    # scores = np.reshape(brain.predict(X_test,Y_test),-1)#predictions
    # scores = brain.predict(X_test)#predictions
    preds = np.reshape(np.argmax(brain.predict(X_test),axis = 1),-1)#predictions
    # print(preds.shape)
    # print(Y_test.shape)
    # print(preds[0],Y_test[0])
    # print('Model {0}: f1 = micro:{1}'.format(i,
    # f1_score(Y_test,preds,average='micro')))
    scores = brain.evaluate(X_test,Y1[test_i])
    print("\n\n{0}: {1:.2f}\n\n\n".format(brain.metrics_names[1], scores[1]*100))
    classes.append(brain)
    # brain.save('Model{0}.h5'.format(i))
    # exit()
    i += 1

    exit()
#save all classes
# for i,m in enumerate(classes):
    # m.save(cur_dir+'\\complex_model\\model{0}.h5'.format(i))

data = read_csv(open('test.csv','r'),na_values='').as_matrix()

X_test = data[:,1:] # input features
# imp = Imputer()#default arguments will suffice
X_test = imp.transform(X_test)

# X_test = X

predictions = vote(classes,X_test)
with open('pred_ANN_ensemble.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))

predictions = vote(classes,X)
print('Model {0}: f1 = micro:{1}  macro:{2}%'.format(i,
f1_score(Y,predictions,average='micro'),
f1_score(Y,predictions,average='macro')))
