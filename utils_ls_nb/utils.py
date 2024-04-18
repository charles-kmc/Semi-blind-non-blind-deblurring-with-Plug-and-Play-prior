# Utils functions used in this project.
# Authors: Charlesquin Kemajou (Heriot-Watt , mbakamkemajou@gmail.com)

import torch
import kornia
import torch.nn as nn
import numpy as np
import math
from scipy import signal
from statsmodels.tsa.stattools import acf as autocorr

# Cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- paaded function for kernel
class Metrices:
    def __init__(self):
        pass
    # --- mse
    def mse_function(self,x,y):
        mse = torch.nn.MSELoss()
        return mse(x,y)

    # --- psnr
    def psnr_function(self,x,y):
        mse = torch.nn.MSELoss()
        return -10*torch.log10(mse(x,y))

    def ess_function(self,arr):
        n = len(arr)
        acf = autocorr(arr, nlags = n, fft = True)
        sums = 0
        for kk in range(1,len(acf)):
            sums = sums + (n-kk)*acf[kk] / n
        return n / (1+2*sums)
        