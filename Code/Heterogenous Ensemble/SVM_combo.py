import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import read_csv
import pickle
from keras.utils.np_utils import to_categorical



data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()

X = data[:,1:-1] # input features
Y = data[:,-1].astype('int') # input features

classes = [
ExtraTreesClassifier(n_estimators=100,n_jobs = -1),
KNeighborsClassifier(n_neighbors=7,p=1,n_jobs=-1),
GaussianNB(),
SVC(kernel='sigmoid'),
SVC(kernel='poly'),
SVC(kernel='rbf'),
]
weights = [0]*len(classes)

try:
    weights = pickle.load(open('S_V_M.pickle','rb'))
except:
    for train_i,test_i in StratifiedKFold(n_splits=2).split(X,Y):
        X_train = X[train_i]
        Y_train = Y[train_i]
        X_test = X[test_i]
        Y_test = Y[test_i]
        for i,cl in zip(range(len(classes)),classes):
            cl.fit(X_train,Y_train)
            sc = cl.score(X_test,Y_test)
            print('Score:',sc)
            weights[i] += sc
        print('K done once')
        break

    print('K done all\n\n')

    # exit()
    try:
        pickle.dump(weights,open('S_V_M.pickle','wb'))
        print('\nDumping Success :}\n')
    except:
        print('\nDumping Failed :(\n')

data = read_csv(open('test_knn.csv','r'),na_values='').as_matrix()
X_test = data[:,1:] # input features


preds = []

for cl,w in zip(classes,weights):
    cl.fit(X,Y)
    print('Train Score:',cl.score(X,Y))
    preds.append(w*to_categorical(np.array(cl.predict(X_test))))

def vote(classes,name):
    votes = np.array(classes)
    print(votes.shape)
    votes = np.sum(votes,axis = 0)
    np.savetxt('net_Probs_{0}_data.txt'.format(name),votes)
    print(votes.shape)
    winners = np.argmax(votes,axis = 1)
    print(winners.shape)
    return winners

predictions = vote(preds,'S_V_M')

with open('S_V_M.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))

