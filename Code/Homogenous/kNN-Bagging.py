import numpy as np
from sklearn.preprocessing import Imputer
from pandas import read_csv
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
    data = read_csv(open('train_pca.csv','r'),na_values='').as_matrix()

    X = data[:,1:-1] # input features
    Y = data[:,-1].astype('int') # input features

    imp = Imputer()#default arguments will suffice
    X = imp.fit_transform(X)

    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, shuffle = Y)

    cl = AdaBoostClassifier(
    base_estimator=KNeighborsClassifier(n_neighbors=5,weights='uniform'),
    n_estimators=5,max_samples=0.6,max_features=0.4,n_jobs=-1,
    bootstrap_features=True, warm_start=True, oob_score=False
    )

    cl.fit(X_train,Y_train)

    predictions = cl.predict(X_train)
    # print(X_train.shape,Y_train.shape,predictions.shape)
    # print(list(zip(Y_train,predictions)))
    print('\n\nModel Train: f1 = micro:{0}  macro:{1}%'.format(
    f1_score(Y_train,predictions,average='micro'),
    f1_score(Y_train,predictions,average='macro')))

    predictions = cl.predict(X_test)
    print('\nModel Test: f1 = micro:{0}  macro:{1}%'.format(
    f1_score(Y_test,predictions,average='micro'),
    f1_score(Y_test,predictions,average='macro')))

    # exit()

    cl.fit(X,Y)

    # pickle.dump(cl,open('kNNhuge_ensemble.pickle','wb'))

    # exit()

    data = read_csv(open('test_pca.csv','r'),na_values='').as_matrix()
    X_test = data[:,1:] # input features
    X_test = imp.transform(X_test)

    predictions = cl.predict(X_test)
    with open('LR_ensemble.csv','w+') as f:
        f.write('id,label\n')
        for i,t in enumerate(predictions):
            f.write('{0},{1}\n'.format(i,t))
