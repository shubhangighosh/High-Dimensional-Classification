import numpy as np
from keras.models import Sequential, load_model
from keras.metrics import categorical_accuracy
from keras.layers import Dense, Activation, Dropout
from pandas import read_csv
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1,l2
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split


data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()

X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features
Y1 = to_categorical(Y)


X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, shuffle = Y)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# print('Verification - Train and Test shapes')
# print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
# print('Verified')
with open('ParamSearch.txt','w+') as f:
    f.write('L1\tD1\tD2\tl2_1\tl2_2\tn_epo\tAccuracy\n\n')
    for L1 in [1300,2600]:
        for D1 in [0,0.2,0.4,0.6]:
            for D2 in [0,0.2,0.4,0.6]:
                for l2_1 in [0,1e-2,1e-1]:
                    for l2_2 in [0,1e-2,1e-1]:
                        for n_epo in [25,60]:
                            brain = Sequential()
                            brain.add(Dense(L1, input_dim = 2600, activation = 'relu', kernel_regularizer=l2(l2_1)))
                            brain.add(Dropout(D1))
                            brain.add(Dense(L1//2 - 7, activation = 'relu', kernel_regularizer=l2(l2_2)))
                            brain.add(Dropout(D2))
                            brain.add(Dense(29, activation = 'softmax'))
                            
                            brain.compile(loss='categorical_crossentropy', optimizer = 'adam',
                             metrics=[categorical_accuracy])
                            brain.fit(X_train,Y_train,epochs=n_epo, batch_size=30, verbose=0, validation_split = 0.2)
                            params = [L1,D1,D2,l2_1,l2_2,n_epo]
                            scores = brain.evaluate(X_test,Y_test)
                            f.write("\n{0}: {1:.2f}".format(params, scores[1]*100))
 