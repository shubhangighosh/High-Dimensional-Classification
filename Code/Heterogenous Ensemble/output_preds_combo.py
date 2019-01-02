from pandas import read_csv
from numpy import *


# mice = read_csv(open('netProbs_mice.txt','r'),header=None,na_values='').as_matrix()[:,1]
# mice = loadtxt('netProbs_mice.txt')
knn = read_csv(open('pred_ANN_ULTRA_ensemble.csv','r'),na_values='').as_matrix()[:,1]
nb = read_csv(open('shu_preds.csv','r'),header=None,na_values='').as_matrix()
rf = read_csv(open('shu_rf.csv','r'),header=None,na_values='').as_matrix()

print(knn.shape,nb.shape,rf.shape)
# exit()
knn = reshape(knn,-1)
nb = reshape(nb,-1)
rf = reshape(rf,-1)
predictions=[]

# print(list(zip(knn,rf,nb)))


for i in range(len(nb)):
    if nb[i] == rf[i]:
        predictions.append(nb[i])
    else:
        predictions.append(knn[i])

with open('combo_preds.csv','w+') as f:
    f.write('id,label\n')
    for i,t in enumerate(predictions):
        f.write('{0},{1}\n'.format(i,t))
