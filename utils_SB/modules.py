

import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

import torchvision.transforms as T

# ---- 
class welford:
    def __init__(self, x, startit = 1):
        self.k = startit
        self.M = x.clone().detach()
        self.S = 0

    def update(self, x):
        with torch.no_grad():
            self.k += 1
            Mnext  =   ((self.k-1) * self.M) / self.k + x / self.k
            self.S = self.S + ((self.k-1) / self.k) * (x - self.M)*(x - self.M)
            self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        return self.S/(self.k-1)

# --- Mathieu
from model.models import *

def load_model(n_ch, n_lev, ljr, device):

    name_file = 'DNCNN_nch_'+str(n_ch)+'_sigma_'+str(n_lev)+'_ljr_'+str(ljr)
    
    model_weights = torch.load('ckpts/finetuned/'+name_file+'.ckpt', map_location=torch.device('cpu'))

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    avg, bn, depth = False, False, 20
    net = DnCNN(1, 1, depth, 'R')

    if cuda:
        model = nn.DataParallel(net).to(device)
    else:
        model = nn.DataParallel(net)

    model.module.load_state_dict(model_weights["state_dict"], strict=True)
    model.eval() 
    
    return model

# ----- Denoiser
def denoiser(y,model):
    ytmp = model(y)
    return ytmp

# --- Ryu
from model.models import *

def load_model_ryu(model_type, sigma, device):
    path = "Pretrained_models/" + model_type + "_noise" + str(sigma) + ".pth"
    if model_type == "DnCNN":
        from model.models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).to(device)
    elif model_type == "SimpleCNN":
        from model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True).to(device)
    elif model_type == "RealSN_DnCNN":
        from model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).to(device)
    elif model_type == "RealSN_SimpleCNN":
        from model.SimpleCNN_models import DnCNN
        model = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True).to(device)
    else:
        from model.realSN_models import DnCNN
        net = DnCNN(channels=1, num_of_layers=17)
        model = nn.DataParallel(net).to(device)

    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# --- Denoiser 

def denoiser_ryu(y, model, eps, r, rotation = True, normalise = True):
    if rotation:
        y = torch.rot90(y, r, [2, 3])
    if normalise:
        mintmp = torch.min(y)
        maxtmp = torch.max(y)
        diff = maxtmp-mintmp
        ytmp = (y.clone() - mintmp)/diff
        scale_range = 1.0 + math.sqrt(eps)/2.0
        scale_shift = (1 - scale_range) / 2.0
        ytmp = ytmp * scale_range + scale_shift
        ytmp -=  model(ytmp)
        # scale and shift the denoised image back
        ytmp = (ytmp - scale_shift) / scale_range
        ytmp = ytmp*diff + mintmp

    else:
        ytmp =  (y - model(y))
    if rotation:
        ytmp = torch.rot90(ytmp, 4-r, [2, 3])
    return ytmp

