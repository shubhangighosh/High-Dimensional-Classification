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
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import os
from keras.layers import Dropout
import pickle
from keras.constraints import maxnorm
from numpy.random import randint
from keras.regularizers import l1,l2



cur_dir = os.path.dirname(__file__)

def vote(classes,name):
    votes = np.array(classes)
    # votes = transpose(votes)
    print(votes.shape)
    try:
        np.savetxt('each_Probs_{0}_data_3D.txt'.format(name),votes)
        np.save(open('each_Probs_{0}_data_3D.npy'.format(name),'wb'),votes)
    except:
        pass
    try:
        v = np.reshape(votes,(-1,29))
        np.savetxt('each_Probs_{0}_data_flattened.txt'.format(name),v)
    except:
        pass
    votes = np.sum(votes,axis = 0)
    # pickle.dump(votes,open(str(cur_dir)+'/netProbs.pickle','w+'))
    np.savetxt('net_Probs_{0}_data.txt'.format(name),votes)
    print(votes.shape)
    # print(votes)
    # winners = np.reshape(mode(votes)[0],-1)
    winners = np.argmax(votes,axis = 1)
    print(winners.shape)
    # print(winners)
    # exit()
    return winners

data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()

X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features
Y1 = to_categorical(Y)

classes = []
voters = []
voters1 = []
# imp = Imputer()#default arguments will suffice
# X = imp.fit_transform(X)


data_t = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()

X_test_data = data_t[:,1:-1] # input features
del data_t
data_t_1 = read_csv(open('test_knn.csv','r'),na_values='').as_matrix()

X_test_data_1 = data_t_1[:,1:] # input features
del data_t_1

# imp = Imputer()#default arguments will suffice
# X_test = imp.transform(X_test)

# X_test = X


i = 0
for _ in range(2):
    for train_i,test_i in StratifiedKFold(n_splits=5,random_state=int(i**1.5+1)).split(X,Y):
        np.random.shuffle(train_i)
        X_train = X[train_i]
        Y_train = Y1[train_i]
        brain = Sequential()
        brain.add(Dense(2600, input_dim = 2600, activation = 'relu', kernel_regularizer=l2(1e-3)))
        brain.add(Dropout(0.2))
        brain.add(Dense(1301, activation = 'relu', kernel_regularizer=l2(1e-4)))
        brain.add(Dropout(0.1))
        brain.add(Dense(29, activation = 'softmax'))
        brain.compile(loss='categorical_crossentropy', optimizer = 'adam',
         metrics=[categorical_accuracy])

        brain.fit(X_train,Y_train,epochs=13, batch_size=30, verbose=0)

        X_test = X[test_i]
        Y_test = Y[test_i]

        # scores = np.reshape(brain.predict(X_test,Y_test),-1)#predictions
        # scores = brain.predict(X_test)#predictions
        # preds = np.reshape(np.argmax(brain.predict(X_test),axis = 1),-1)#predictions
        # print(preds.shape)
        # print(Y_test.shape)
        # print(preds[0],Y_test[0])
        # print('Model {0}: f1 = {1}'.format(i,
        # f1_score(Y_test,preds,average='micro')))
        scores = brain.evaluate(X_test,Y1[test_i])
        print("\n\n{0}\n{1}: {2:.2f}\n\n\n".format(i,brain.metrics_names[1], scores[1]*100))
        # classes.append(brain)
        voters.append(brain.predict(X_test_data))
        voters1.append(brain.predict(X_test_data_1))
        i += 1

#save all classes
# for i,m in enumerate(classes):
    # m.save(cur_dir+'\\complex_model\\model{0}.h5'.format(i))

predictions = vote(voters,'train')
predictions = vote(voters1,'test')

"""
with open('pred_ANN_ensemble_last_push.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))
predictions = mem_vote(voters)
with open('pred_ANN_ensemble_pca.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))
"""
# predictions = vote(classes,X)
# print('Model {0}: f1 = micro:{1}  macro:{2}%'.format(i,
# f1_score(Y,predictions,average='micro'),
# f1_score(Y,predictions,average='macro')))
