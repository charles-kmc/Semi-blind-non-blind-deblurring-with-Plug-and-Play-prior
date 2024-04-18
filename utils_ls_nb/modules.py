

import torch
import torch.nn as nn
import kornia
import matplotlib.pyplot as plt
import numpy as np
import math

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

# ----- Denoiser
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
        ytmp -=  model(ytmp.float())
        # scale and shift the denoised image back
        ytmp = (ytmp - scale_shift) / scale_range
        ytmp = ytmp*diff + mintmp

    else:
        ytmp =  (y.float() - model(y.float() ))
    if rotation:
        ytmp = torch.rot90(ytmp, 4-r, [2, 3])
    return ytmp


# --- Mathieu
class DnCNN_m(nn.Module):
    def __init__(self, nc_in, nc_out, depth, act_mode, bias=True, nf=64):
        super(DnCNN_m, self).__init__()

        self.depth = depth

        self.in_conv = nn.Conv2d(nc_in, nf, kernel_size=3, stride=1, padding=1, bias=bias)
        self.conv_list = nn.ModuleList(
            [nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias) for _ in range(self.depth - 2)])
        self.out_conv = nn.Conv2d(nf, nc_out, kernel_size=3, stride=1, padding=1, bias=bias)

        if act_mode == 'R':  # Kai Zhang's nomenclature
            self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

    def forward(self, x_in):

        x = self.in_conv(x_in)
        x = self.nl_list[0](x)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x)
            x = self.nl_list[i + 1](x_l)

        return self.out_conv(x) + x_in
def load_model_mathieu(n_ch, n_lev, ljr, device):

    name_file = 'DNCNN_nch_'+str(n_ch)+'_sigma_'+str(n_lev)+'_ljr_'+str(ljr)
    
    model_weights = torch.load('ckpts_mathieu/finetuned/'+name_file+'.ckpt', map_location=torch.device('cpu'))

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    avg, bn, depth = False, False, 20
    net = DnCNN_m(1, 1, depth, 'R')

    if cuda:
        model = nn.DataParallel(net).to(device)
    else:
        model = nn.DataParallel(net)

    model.module.load_state_dict(model_weights["state_dict"], strict=True)
    model.eval() 
    
    return model

# ----- Denoiser
def denoiser_m(y,model,r, rotation = True):
    if rotation:
        y = torch.rot90(y, r, [2, 3])
    ytmp =  model(y.float())
    if rotation:
        ytmp = torch.rot90(ytmp, 4-r, [2, 3])
    return ytmp


def projbox(x, lower,upper):
    return torch.clamp(x, min = lower, max=upper)

def actualconvolve(input, kernel, flipkern = False):
    if flipkern:
        return(kornia.filters.filter2d(input,kernel.rot90(2,[2,1]), border_type='circular'))
    else:
        return(kornia.filters.filter2d(input,kernel, border_type='circular'))


# Welford's algorithm for calculating mean and variance
# https://doi.org/10.2307/1266577
    
class welford:
    def __init__(self, x, startit = 1):
        self.k = startit
        self.M = x.clone().detach()
        self.S = 0

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S += (x - self.M)*(x - Mnext)
        self.M = Mnext
    
    def get_mean(self):
        return self.M
    
    def get_var(self):
        return self.S/(self.k-1)

class Metrics():
    def __init__(self) -> None:
        pass
    # --- mse
    def mse_function(self,x,y):
        mse = torch.nn.MSELoss()
        return mse(x,y)

    # --- psnr
    def psnr_function(self,x,y):
        msexy = self.mse_function(x,y)
        return -10*torch.log10(msexy)