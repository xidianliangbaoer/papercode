# 5000
import os
import math
import torch as t
import numpy as np
import pandas as pd
from scipy import signal
from torch.utils import data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
class Emotion(data.Dataset):
    def __init__(self,root,train = True,K=5,k=0):
        # K 指一共做多少折的交叉验证
        # k表示当前是第几折
        self.train = train

        X_shuffle = pd.read_csv('X_shuffle_bcg.csv')['fname'].values
        print("load shuffle")
        fold_num = int(len(X_shuffle)/K)
        X_test = X_shuffle[fold_num * k:fold_num * (k + 1)]
        if(k == 0):
            X_train = X_shuffle[fold_num * (k + 1)::]
        else:
            temp1 = X_shuffle[0:fold_num * k]
            temp2 = X_shuffle[fold_num * (k + 1)::]
            X_train = np.concatenate([temp1,temp2],axis=0)
        label0 = 0
        label1 = 0
        label2 = 0
        for item in X_train:
            l = item.split('_')[-1]
            if (l[0] == 'P'):
                label0 = label0+1
            elif (l[0] == 'N'):
                label1 = label1+1
            else:
                label2 = label2+1
        print("label0",label0)
        print("label1", label1)
        print("label2", label2)
        df_0 = pd.DataFrame(X_train,columns=['train_fname'])
        df_1 = pd.DataFrame(X_test,columns=['test_fname'])
        df = pd.concat([df_0,df_1],axis=1)
        df.to_csv('logitsave/data_train_test.csv')
        if self.train:
            self.ecgData = [os.path.join(root,hs) for hs in X_train]
        else:
            
            self.ecgData = [os.path.join(root,hs) for hs in X_test]
    def __len__(self):
        return len(self.ecgData)
    def __getitem__(self, item):
        esgPath = self.ecgData[item]

        temp  = esgPath.split('/')[-1].split('.')[0].split('_')[-1]
        # print(temp[0])
        if (temp[0]=='P'):
            label = 0
        elif(temp[0]=='N'):
            label = 1
        else:
            label = 2
        # print(esgPath)
        data = np.load(esgPath, allow_pickle=True)
        data = Minmax(data)

        return data,label

def Minmax(sig):
    sig = StandardScaler().fit_transform(np.array(sig).reshape(-1, 1))
    sig = t.tensor(sig,dtype=t.float)
    return sig

if __name__ == '__main__':
    root = r'../30s_slide0_new'

    ecg = pd.read_csv('fname_30s_slide0_new.csv').values[:, 1]
    laf = np.ones(len(ecg))
    X_train, X_test, _, _ = train_test_split(ecg, laf, test_size=0.2, random_state=0, shuffle=True)

    print("end")
