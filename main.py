
# 可以连续把gama的值跑一遍
import torch
import torch.nn as nn
from torch.optim import Adam,SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

import models

from dataset_kfold import Emotion


from functions import *
from focalloss import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# convnet151、
num_epoch = 300
num_class = 3
batch_size = 64


learning_rate = 0.0001
wd = 0.001
momentum = 0.9

root = r'../filter_30s_slde0_bcg'

savepath = 'modelsave21_4/'
modelname = 'SECnn'
result_train = []
result_test = []


features_extractors_path = None
Classifier_path = None

def train(features_extractors,classifier,data_src,criterion,optimizer,epoch,k,gama):
    correct = 0
    total = 0
    lossTrain = 0
    features_extractors.train()
    classifier.train()
    for i, (data, labels) in enumerate(data_src):
        data = data.reshape([data.shape[0], 1, data.shape[1]])
        data = data.type(torch.cuda.FloatTensor)
        data = data.to(device)
        labels = labels.to(device)
        outputs = classifier(features_extractors(data))
        if gama==0:
            loss = criterion(outputs, labels)
        else:
            # gama=0.5、1、2、3、4
            loss = FocalLoss(gamma=gama)(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        lossTrain = lossTrain + loss.item()
    lossTrain = lossTrain / len(data_src)
    acc = correct * 100. / len(data_src.dataset)
    res_e = 'FOLD:{},Gama:{}, Epoch: [{}/{}], training loss: {:.6f}, correct: [{}/{}], training accuracy: {:.4f}%'.format(
        k,gama,epoch, num_epoch, lossTrain, correct, len(data_src.dataset), acc)
    tqdm.write(res_e)
    log_train.write(res_e + '\n')
    result_train.append([epoch, lossTrain, acc])
    return features_extractors,classifier
def eval(features_extractors,classifier,data_loader,epoch,k,gama):
    features_extractors.eval()
    classifier.eval()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    confusion_matrix = ConfusionMeter(num_class, normalized=False)
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_loader):
            data = data.reshape([data.shape[0], 1, data.shape[1]])
            data = data.type(torch.cuda.FloatTensor)
            data = data.to(device)
            label = target.to(device)
            features = features_extractors(data)
            class_output = classifier(features)
            _, pred = torch.max(class_output.data, 1)
            correct += (pred == label).sum().item()
            loss = criterion(class_output, label)
            total_loss = total_loss + loss.item()
            confusion_matrix.add(F.softmax(class_output,dim=1),label)
        total_loss = total_loss / len(data_loader)
        acc = correct * 100 / len(data_loader.dataset)

        res_e = 'FOLD:{},Gama:{},Epoch: [{}/{}],eval loss: {:.6f}, correct: [{}/{}], accuracy: {:.4f}%,confusion_matrix: {}'.format(
            k,gama,epoch, num_epoch,total_loss, correct, len(data_loader.dataset), acc,confusion_matrix.value())

        tqdm.write(res_e)
        log_test.write(res_e + '\n')
        result_test.append([epoch, total_loss, acc])
    features_extractors.train()
    classifier.train()
    # return total_loss,acc
    return total_loss, acc, confusion_matrix.value()
def predict(features_extractors,classifier,data_loader):
    features_extractors.eval()
    classifier.eval()
    label_all=[]
    predict=[]

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(data_loader):
            data = data.reshape([data.shape[0], 1, data.shape[1]])
            data = data.type(torch.cuda.FloatTensor)
            data = data.to(device)
            label = target.to(device)
            features = features_extractors(data)
            class_output = classifier(features)
            _, pred = torch.max(class_output.data, 1)
            label_all.append(label)
            predict.append(pred)
    features_extractors.train()
    classifier.train()
    label_all = torch.cat(label_all, dim=0).cpu().numpy()
    predict = torch.cat(predict, dim=0).cpu().numpy()
    return label_all,predict

if __name__ == '__main__':

    torch.manual_seed(1)
    # K = 5
    # k = 0

    K = [0,1,2,3,4]
    gamas = [3]
    # for k in range(K):
    for k in K:
        print('交叉验证', k)
        # if k==1:
        #     gamas = [2,3]
        # else:
        #     gamas = [0.5, 0, 1, 2, 3]
        res_e = '交叉验证' + str(k)
        train_dataset = Emotion(root, train=True, K=5, k=k)
        data_src = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=1, drop_last=False)

        test_dataset = Emotion(root, train=False, K=5, k=k)
        data_tar = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)
        for gama in gamas:
            acc_max = 80
            result_train = []
            result_test = []
 
            log_train = open(savepath + modelname + '_fold_' + str(k)+'_gama_'+str(gama) + '_log_train_s-t.txt', 'w')
            log_test = open(savepath + modelname + '_fold_' + str(k) +'_gama_'+str(gama)+ '_log_test_s-t.txt', 'w')
            log_train.write(res_e + '\n')
            features_extractors = init_model(modelname, features_extractors_path, device)
            classifier_name = 'Classifier_conv15'
            classifier = init_model(classifier_name, Classifier_path, device)
            criterion = nn.CrossEntropyLoss()
            parameters_features_extractors = features_extractors.get_parameters()
            parameters_classifier = classifier.get_parameters()
            parameters = parameters_features_extractors + parameters_classifier
            optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=wd)
            # optimizer = SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=wd, nesterov=True)
            # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            confusion_matrix = ConfusionMeter(num_class, normalized=False)
            info = dict()
            info['id'] = k
            for epoch in tqdm(range(1, num_epoch + 1)):
                features_extractors, classifier = train(features_extractors, classifier, data_src, criterion, optimizer,
                                                        epoch, k,gama)
                total_loss, acc,current_cm = eval(features_extractors, classifier, data_tar, epoch, k,gama)

                if (epoch % 40 == 0):
                # if (epoch % 20 == 0):

                    for p in optimizer.param_groups:
                        p['lr'] *= 0.9
                #         # p['lr'] *= pow(0.9, 6)

                if (epoch % 10 == 0):
                    res_train = np.asarray(result_train)
                    res_test = np.asarray(result_test)
                    path1 = savepath + modelname + '_fold_' + str(k)+'_gama_'+str(gama) + '_res_train_a-w.csv'
                    np.savetxt(path1, res_train, fmt='%.6f', delimiter=',')
                    path2 = savepath + modelname + '_fold_' + str(k)+'_gama_'+str(gama) + '_res_test_a-w.csv'
                    np.savetxt(path2, res_test, fmt='%.6f', delimiter=',')
                if (epoch == num_epoch):
                    fname = savepath + modelname + '_' + 'fold_' + str(k)+'_gama_'+str(gama) + '_tsne.png'
                    src_feature, src_label = collect_feature(data_src, features_extractors, device)
                    tar_feature, tar_label = collect_feature(data_tar, features_extractors, device)
                    visualize(src_feature, tar_feature, filename=fname, src_label=src_label, tar_label=tar_label)
                    model_name = savepath + modelname + '_fold_' + str(k)+'_gama_'+str(gama) + '.pth'
                    features_extractors.save(name=model_name)
                    model_name = savepath + modelname + classifier_name + '_fold_' + str(k) +'_gama_'+str(gama)+ '.pth'
                    classifier.save(name=model_name)



            log_train.close()
            log_test.close()



