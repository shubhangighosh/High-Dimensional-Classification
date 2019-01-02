import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle

if __name__ == '__main__':
    data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()

    X = data[:,1:-1] # input features
    Y = data[:,-1].astype('int') # input features

    pca = PCA(n_components = 0.95, whiten=True)
    scale = StandardScaler()

    X = pca.fit_transform(X)
    X = scale.fit_transform(X)

    train = np.concatenate((X,np.reshape(Y,(-1,1))),axis=1)
    print('#Components: ',pca.n_components_)
    headers = list(['f_{0}'.format(i) for i in range(pca.n_components_)])

    pd.DataFrame(train).to_csv('train_pca.csv', header = headers + ['label'], index_label='id')
    print(X.shape)


    # pickle.dump(pca,open('kNNhuge_ensemble.pickle','wb'))

    # exit()

    data = read_csv(open('test_knn.csv','r'),na_values='').as_matrix()
    X_test = data[:,1:] # input features
    X_test = pca.transform(X_test)
    X_test = scale.transform(X_test)
    print(X_test.shape)
    pd.DataFrame(X_test).to_csv('test_pca.csv', header = headers, index_label='id')
    
