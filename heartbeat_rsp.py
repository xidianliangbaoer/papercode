
import h5py
import pywt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False
from scipy import signal
import os

def xiaobo(data):
 
    mode = pywt.Modes.smooth
    w = pywt.Wavelet('db4')  
    a = data
    ca = []  
    cd = []  
    for i in range(9):
        (a, d) = pywt.dwt(a, w, mode)  
        ca.append(a)
        cd.append(d)
    coeff_list = [None]
    for i in range(5):
        coeff_list.append(cd[6 - i])
    coeff_list.append(None)
    coeff_list.append(None)
    aaa = pywt.waverec(coeff_list, w)  
    return aaa
def xiaobo(data):

    fs = 250
    f1 = 3
    f2 = 24
    w1 = f1 / fs * 2
    w2 = f2 / fs * 2
    b, a = signal.butter(3, [w1, w2], 'bandpass')
    filtedData_ori = signal.filtfilt(b, a, data)
    bcg0 = filtedData_ori
    plt.plot(filtedData_ori)
    plt.title("原始信号滤波")
    plt.show()
