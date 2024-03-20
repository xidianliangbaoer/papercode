from .BasicModule import BasicModule
from torch import nn
import torch as t
from torch.nn import functional as F
from typing import Tuple, Optional, List, Dict

class AlexNet(BasicModule):
    def __init__(self):
        super(AlexNet,self).__init__()

        self.feature = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=64,kernel_size=11,stride=4,padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),

            nn.Conv1d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(6),
        )
    def get_parameters(self,base_lr = 1)-> List[Dict]:
        params =[
            {'params':self.parameters()}
        ]
        return params
    def forward(self,x):
        x =self.feature(x)
        x = x.view(-1,1536)
        return x