from sklearn.manifold import MDS
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

lst = ["f_0","f_1","f_2","f_3","f_4","f_5","f_6","f_7","f_8","f_9",
"f_10","f_11","f_12","f_13","f_14","f_15","f_16","f_17","f_18","f_19",
"f_20","f_21","f_22","f_23","f_24","f_25","f_26","f_27","f_28","f_29",
"f_30","f_31","f_32","f_33","f_34","f_35","f_36","f_37","f_38","f_39",
"f_40","f_41","f_42","f_43","f_44","f_45","f_46","f_47","f_48","f_49",
"f_50","f_51","f_52","f_53","f_54","f_55","f_56","f_57","f_58","f_59",
"f_60","f_61","f_62","f_63","f_64","f_65","f_66","f_67","f_68","f_69",
"f_70","f_71","f_72","f_73","f_74","f_75","f_76","f_77","f_78","f_79",
"f_80","f_81","f_82","f_83","f_84","f_85","f_86","f_87","f_88","f_89",
"f_90","f_91","f_92","f_93","f_94","f_95","f_96","f_97","f_98","f_99",
"f_100","f_101","f_102","f_103","f_104","f_105","f_106","f_107","f_108",
"f_109","f_110","f_111","f_112","f_113","f_114","f_115","f_116","f_117",
"f_118","f_119","f_120","f_121","f_122","f_123","f_124","f_125","f_126",
"f_127","f_128","f_129","f_130","f_131","f_132","f_133","f_134","f_135",
"f_136","f_137","f_138","f_139","f_140","f_141","f_142","f_143","f_144",
"f_145","f_146","f_147","f_148","f_149","f_150","f_151","f_152","f_153",
"f_154","f_155","f_156","f_157","f_158","f_159","f_160","f_161","f_162",
"f_163","f_164","f_165","f_166","f_167","f_168","f_169","f_170","f_171",
"f_172","f_173","f_174","f_175","f_176","f_177","f_178","f_179","f_180",
"f_181","f_182","f_183","f_184","f_185","f_186","f_187","f_188","f_189",
"f_190","f_191","f_192","f_193","f_194","f_195","f_196","f_197","f_198",
"f_199","f_200","f_201","f_202","f_203","f_204","f_205","f_206","f_207",
"f_208","f_209","f_210","f_211","f_212","f_213","f_214","f_215","f_216",
"f_217","f_218","f_219","f_220","f_221","f_222","f_223","f_224","f_225",
"f_226","f_227","f_228","f_229","f_230","f_231","f_232","f_233","f_234",
"f_235","f_236","f_237","f_238","f_239","f_240","f_241","f_242","f_243",
"f_244","f_245","f_246","f_247","f_248","f_249","f_250","f_251","f_252",
"f_253","f_254","f_255","f_256","f_257","f_258","f_259",
"label"]

if True:
    data = read_csv(open('train_knn.csv','r'),na_values='').as_matrix()
    X1 = data[:,1:-1] # input features
    train_i = X1.shape[0]
    Y1 = data[:,-1].astype('int') # input features
    del data


    data = read_csv(open('test_knn.csv','r'),na_values='').as_matrix()
    X2 = data[:,1:] # features
    del data

    tr = MDS(n_components = 260, verbose = 1, max_iter = 20)
    sc = StandardScaler()
    # print(X1.shape,np.transpose(Y1).shape)

    
    X_net = np.concatenate((X1,X2))

    X_net = tr.fit_transform(X_net)
    X_net = sc.fit_transform(X_net)

    X1 = X_net[:train_i]
    test = X_net[train_i:]
    del X_net

    train = np.concatenate((X1,np.reshape(Y1,(-1,1))),axis=1)
    del Y1,X1
    
    pd.DataFrame(train).to_csv('train_MDS.csv', header = lst,index_label ='id')
    del train

    # print('Train done:',train.shape,data.shape)

    # pd.DataFrame(X_net).to_csv('all.csv', header = None)
    # print(train.shape,test.shape)

    pd.DataFrame(test).to_csv('test_MDS.csv', header = lst[:-1],index_label ='id')

    # print('Test done:',test.shape,data.shape)
    del test


# data = read_csv(open('test.csv','r'),na_values='').as_matrix()
# test = np.concatenate((np.reshape(np.arange(2900),(-1,1)),test),axis=1)
