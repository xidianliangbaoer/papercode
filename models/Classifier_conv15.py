from torch import nn
from .BasicModule import BasicModule
from typing import Tuple, Optional, List, Dict
class Classifier_conv15(BasicModule):
    def __init__(self,input_size = 32,hid_size = 32,numclass = 3):
        super(Classifier_conv15,self).__init__()
        self.class_classifier1 = nn.Sequential()
        self.class_classifier1.add_module('c_fc1',nn.Linear(input_size*12, hid_size*6))
        self.class_classifier1.add_module('c_LeakyReLU1', nn.LeakyReLU())
        self.class_classifier2 = nn.Sequential()
        self.class_classifier2.add_module('c_drop1', nn.Dropout(p=0.5))
        self.class_classifier2.add_module('c_fc2', nn.Linear(input_size*6, hid_size*3))
        self.class_classifier2.add_module('c_LeakyReLU2', nn.LeakyReLU())
        self.class_classifier2.add_module('c_drop2', nn.Dropout(p=0.5))
        self.class_classifier2.add_module('c_fc3', nn.Linear(hid_size * 3, numclass))

        # self.class_classifier1 = nn.Sequential()
        # self.class_classifier1.add_module('c_fc1',nn.Linear(input_size*8, hid_size*4))
        # self.class_classifier1.add_module('c_LeakyReLU1', nn.LeakyReLU())
        # self.class_classifier2 = nn.Sequential()
        # self.class_classifier2.add_module('c_drop1', nn.Dropout(p=0.1))
        # self.class_classifier2.add_module('c_fc2', nn.Linear(hid_size*4, numclass))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def get_parameters(self, base_lr=1) -> List[Dict]:
        params = [
            {'params': self.parameters()}
        ]
        return params

    def forward(self,features):
        features = self.class_classifier1(features)
        class_output = self.class_classifier2(features)
        return class_output
