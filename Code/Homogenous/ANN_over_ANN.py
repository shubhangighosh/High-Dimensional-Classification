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
from sklearn.model_selection import StratifiedKFold
import pickle
import os
cur_dir = os.path.dirname(__file__)


def vote(classes):
    votes = np.array(classes)
    # votes = transpose(votes)
    print(votes.shape)
    votes = np.sum(votes,axis = 0)
    # pickle.dump(votes,open(str(cur_dir)+'/netProbs.pickle','w+'))
    np.savetxt('netProbs_ANN_ANN.txt',votes)
    print(votes.shape)
    # print(votes)
    # winners = np.reshape(mode(votes)[0],-1)
    winners = np.argmax(votes,axis = 1)
    print(winners.shape)
    # print(winners)
    # exit()
    return winners


# data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()
# X = read_csv(open('best_train.csv','r'),header=None,na_values='').as_matrix()
X = np.loadtxt('here_now.txt')
print(X.shape)
#X  = X / np.sum(X) * X.shape[0]


# data_t = read_csv(open('probs_knn_test.csv','r'),header=None,na_values='').as_matrix()
X_test_data = np.loadtxt('test_ds.txt')


X_test_data = np.log(X_test_data*np.sum(X) / np.sum(X_test_data))
X = np.log(X)

try:
    Y = pickle.load(open('Ys.pickle','rb'))
except:
    Y = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()[:,-1].astype('int') # input features
    pickle.dump(Y,open('Ys.pickle','wb'))
Y1 = to_categorical(Y)

classes = []
voters = []
# Dropout(rate, noise_shape=None, seed=None)
avg = []
i = 0
for _ in range(10):
    for train_i,test_i in StratifiedKFold(n_splits=3,random_state=(i+3)**2).split(X,Y):
        np.random.shuffle(train_i)
        X_train = X[train_i]
        Y_train = Y1[train_i]
        brain = Sequential()
        brain.add(Dense(29, input_dim = 29, activation = 'relu', kernel_regularizer=l2(1e-4)))
        # brain.add(Dropout(0.1))
        brain.add(Dense(55, activation = 'relu', kernel_regularizer=l2(1e-4)))
        # brain.add(Dense(1197, activation = 'relu', kernel_constraint=maxnorm(5), kernel_regularizer=l2(0.01)))
        brain.add(Dropout(0.05))
        brain.add(Dense(29, activation = 'softmax'))
        brain.compile(loss='categorical_crossentropy', optimizer = 'adam',
         metrics=[categorical_accuracy])

        brain.fit(X_train,Y_train,epochs=60, batch_size=30, verbose=0, validation_split = 0.2)

        X_test = X[test_i]
        Y_test = Y[test_i]
        voters.append(brain.predict(X_test_data))
        preds = np.reshape(np.argmax(brain.predict(X_test),axis = 1),-1)#predictions

        scores = brain.evaluate(X_test,Y1[test_i])
        print("\n\n{0}: {1:.2f}\n\n\n".format(brain.metrics_names[1], scores[1]*100))

        avg.append(scores[1])
        i += 1

print('Average:',np.average(avg)*100)
# exit()

predictions = vote(voters)
with open('pred_ANN_oh_ANN.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))


