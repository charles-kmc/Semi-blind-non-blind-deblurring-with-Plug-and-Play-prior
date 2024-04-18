import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import os
import math
from collections import defaultdict
from scipy.stats.mstats import mquantiles
import pandas
import scipy.io
import cv2

import torch
import kornia

import time as time
from tqdm.auto import tqdm
import datetime
import sys

# Cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inpaint_operator(rho, im_shape):
    '''
    Inputs:
        - rho (float <1): proportion of pixels damaged by the operator
        - im_shape (vector): Shape of the damaging operator.
    Output:
        - Omega (matrix): Damaging operator.
    '''
    # image size
    N = torch.prod(im_shape)
    Omega = torch.zeros(im_shape)
    
    # random permutation 
    sel = random.permutation(N)
    
    # pixels that the operator will damage
    ind = sel[torch.arange(int(rho*N))]
    
    # Damaging operator
    torch.ravel(Omega)[ind]
    
    return 1 - Omega
