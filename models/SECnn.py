from .BasicModule import BasicModule
from torch import nn
import torch as t
from torch.nn import functional as F
# from functions import ReverseLayerF
from typing import Tuple, Optional, List, Dict
device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
class SECnn(BasicModule):
    def __init__(self,kernel_size=32,hid_size=32,channel =32 * 12,reduction =16):
        # reduction =16
        super(SECnn,self).__init__()
        self.hidden_size = 100
        self.num_layers = 2
        self.feature = nn.Sequential()
        # cnn block1
        self.feature.add_module('f_conv1',nn.Conv1d(in_channels=1, out_channels=hid_size, kernel_size=kernel_size, stride=1))
        self.feature.add_module('f_bn1',nn.BatchNorm1d(32))
        self.feature.add_module('f_LeakyReLU1',nn.LeakyReLU())
        self.feature.add_module('f_pool1',nn.MaxPool1d(kernel_size=8, stride=2))
        # cnn block2
        self.feature.add_module('f_conv2',nn.Conv1d(in_channels=hid_size, out_channels=hid_size*2, kernel_size=kernel_size//2, stride=1))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(64))
        self.feature.add_module('f_LeakyReLU2', nn.LeakyReLU())
        self.feature.add_module('f_pool2',nn.MaxPool1d(kernel_size=8, stride=2))
        # cnn block3
        self.feature.add_module('f_conv3',nn.Conv1d(in_channels=hid_size*2, out_channels=hid_size*4, kernel_size=kernel_size//4, stride=1))
        self.feature.add_module('f_bn3', nn.BatchNorm1d(128))
        self.feature.add_module('f_LeakyReLU3', nn.LeakyReLU())
        self.feature.add_module('f_pool3',nn.MaxPool1d(kernel_size=8, stride=2))
        # cnn block4
        self.feature.add_module('f_conv4',nn.Conv1d(in_channels=hid_size*4, out_channels=hid_size*4, kernel_size=kernel_size//4, stride=1))
        self.feature.add_module('f_bn4', nn.BatchNorm1d(128))
        self.feature.add_module('f_LeakyReLU4', nn.LeakyReLU())
        self.feature.add_module('f_pool3',nn.MaxPool1d(kernel_size=8, stride=2))
        # cnn block5
        self.feature.add_module('f_conv5',nn.Conv1d(in_channels=hid_size*4, out_channels=hid_size*4, kernel_size=kernel_size//4, stride=1))
        self.feature.add_module('f_bn5', nn.BatchNorm1d(128))
        self.feature.add_module('f_LeakyReLU5', nn.LeakyReLU())
        self.feature.add_module('f_pool5',nn.MaxPool1d(kernel_size=8, stride=2))
        # cnn block6
        self.feature.add_module('f_conv6',nn.Conv1d(in_channels=hid_size*4, out_channels=hid_size*8, kernel_size=kernel_size//8, stride=1))
        self.feature.add_module('f_bn6', nn.BatchNorm1d(hid_size*8))
        self.feature.add_module('f_LeakyReLU6', nn.LeakyReLU())
        self.feature.add_module('f_pool6',nn.MaxPool1d(kernel_size=4, stride=2))
        # cnn block7
        self.feature.add_module('f_conv7',nn.Conv1d(in_channels=hid_size*8, out_channels=hid_size*12, kernel_size=kernel_size//8, stride=1))
        self.feature.add_module('f_bn7', nn.BatchNorm1d(hid_size*12))
        self.feature.add_module('f_LeakyReLU7', nn.LeakyReLU())

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.feature1 = nn.Sequential()
        self.feature1.add_module('f_pool',nn.AdaptiveAvgPool1d(1))

        # self.feature1.add_module('f_LeakyReLU1', nn.LeakyReLU())
        self.out_dim = hid_size * 12
    def get_parameters(self,base_lr = 1)-> List[Dict]:
        params =[
            {'params':self.parameters()}
        ]
        return params

    def forward(self,input_data):
        h0 = t.randn(self.num_layers * 2, input_data.size(0), self.hidden_size).to(device)  # 2 for bidirection
        features = self.feature(input_data)
        # print(features.size())
        # features = features.reshape(features.size(0), -1)
        y = self.avg_pool(features).view(features.size(0),-1)
        # print(y.size())
        y = self.fc(y).view(features.size(0),32*12,-1)
        features = features*y.expand_as(features)
        features = self.feature1(features)
        features = features.reshape(features.size(0), -1)
        # print(features.size())
        return features
