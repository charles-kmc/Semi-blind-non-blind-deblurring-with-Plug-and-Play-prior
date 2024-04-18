
import torch
import torch.nn as nn
import numpy as np
import math
from scipy import signal
from statsmodels.tsa.stattools import acf as autocorr
from utils_SB.utils import *

# --- Gaussian filter
def Gaussian_filter(taille, a, b):
    
    center = int((taille+1)/ 2)
    x = np.arange(-taille + center, taille - center+1)
    y = np.arange(-taille + center, taille - center+1)
    
    [v,u] = np.meshgrid(x,y)
    
    c = a**2 * u**2 + b**2 * v**2
    kernel = ((a * b) / (2*np.pi)) * np.exp(-c/2)
    kernel = kernel / np.sum(kernel.reshape(-1,1))
    
    return torch.from_numpy(kernel)

# -- Sum for the normilisation constant
def Sum_gauss_psf(taille, a,b):
    
    center = int((taille+1)/ 2)
    x = np.arange(-taille + center, taille - center+1)
    y = np.arange(-taille + center, taille - center+1)
    
    [v,u] = np.meshgrid(x,y)
    
    c     = a**2 * u**2 + b**2 * v**2
    f     = ((a * b) / (2*np.pi)) * np.exp(-c/2)
    diffb = (a / (2*np.pi)) * (1 - b**2 * v**2) * np.exp(-c / 2)
    diffa = (b / (2*np.pi)) * (1 - a**2 * u**2) * np.exp(-c / 2)
    
    sum_f     = np.sum(f.reshape(-1,1))
    sum_diffa = np.sum(diffa.reshape(-1,1))
    sum_diffb = np.sum(diffb.reshape(-1,1))
    
    return sum_f, sum_diffa, sum_diffb
    
# --- Gradient
def diff_kernel_a(taille, a,b, x_shape):
    
    center = int((taille+1)/ 2)
    x = np.arange(-taille + center, taille - center+1)
    y = np.arange(-taille + center, taille - center+1)
    
    [v,u] = np.meshgrid(x,y)
    sum_f, sum_diffa, _ = Sum_gauss_psf(taille, a, b)
    
    c = a**2 * u**2 + b**2 * v**2
    f = ((a * b) / (2*np.pi)) * np.exp(-c/2)
    diffa = (b / (2*np.pi)) * (1 - a**2 * u**2) * np.exp(-c / 2)
    diffa_kernel = (diffa * sum_f - f * sum_diffa) / (sum_f**2)
    rz_diffa_kernel = resize_im(torch.from_numpy(diffa_kernel), x_shape)
    
    return rz_diffa_kernel

# --- Gradient
def diff_kernel_b(taille, a, b, x_shape):
    
    center = int((taille+1)/ 2)
    x = np.arange(-taille + center, taille - center+1)
    y = np.arange(-taille + center, taille - center+1)
    
    [v,u] = np.meshgrid(x,y)
    sum_f, _, sum_diffb = Sum_gauss_psf(taille, a,b)
    
    c = a**2 * u**2 + b**2 * v**2
    f = ((a * b) / (2*np.pi)) * np.exp(-c/2)
    diffb = (a / (2*np.pi)) * (1 - b**2 * v**2) * np.exp(-c / 2)
    diffb_kernel = (diffb * sum_f - f * sum_diffb) / (sum_f**2)
    rz_diffb_kernel = resize_im(torch.from_numpy(diffb_kernel), x_shape)
    
    return rz_diffb_kernel
    
######## --- Moffat filter

def Moffat_filter(taille,a,b):
    center = int((taille+1)/ 2)
    x = np.arange(-taille + center, taille - center+1)
    y = np.arange(-taille + center, taille - center+1)
    
    [v,u] = np.meshgrid(x,y)
    
    b_2 = b+2
    
    a2 = np.power(a,2)
                        
    r2 = np.power(u,2) + np.power(v,2)
    
    val = 1 + (r2*a2) / b
    
    kernel =  (a2 / (2*np.pi)) * np.power(val,-b_2/2)
    
    kernel = kernel / np.sum(kernel)

    return torch.from_numpy(kernel)  

# --- gradient to alpha
def grad_moffat_alpha(taille, a, b, x_shape):
    
    sum_kernel_mof,sum_grad_alpha, _ = sum_mof_filter(taille, a, b)
    
    center = int((taille+1)/ 2)
    
    X = np.arange(-taille + center, taille - center+1)
    Y = np.arange(-taille + center, taille - center+1)
    
    grad_alpha = np.zeros((taille, taille))
    
    for ii in range(taille):
        for jj in range(taille):
            
            b_2 = b+2
            
            a2 = np.power(a,2)
                        
            r2 = np.power(X[ii],2) + np.power(Y[jj],2)
            
            val = 1 + (r2*a2) / b
            
            f =  a2 * np.power(val,-b_2/2) / (2*np.pi)
            
            grad = ((2 - ((b_2*r2*a2) / (b + r2*a2))) * np.power(val,-b_2/2)) * (a / (2*np.pi))
            
            grad_alpha[ii,jj] = (grad * sum_kernel_mof - f * sum_grad_alpha) / (sum_kernel_mof)**2
    
    rz_grad = resize_im(torch.from_numpy(grad_alpha), x_shape)
    return  rz_grad 

# --- gradient beta
def grad_moffat_beta(taille, a, b, x_shape):
    sum_kernel_mof, _, sum_grad_beta = sum_mof_filter(taille, a, b)
    center = int((taille+1)/ 2)
    X = np.arange(-taille + center, taille - center+1)
    Y = np.arange(-taille + center, taille - center+1)
    grad_beta = np.zeros((taille, taille))
    for ii in range(taille):
        for jj in range(taille):
            b_2 = b+2
            
            a2 = np.power(a,2)
            
            cons1 = a2 / (4*np.pi)
            
            r2 = np.power(X[ii],2) + np.power(Y[jj],2)
            
            val = 1 + (r2*a2) / b
            
            f =  (a2 / (2*np.pi)) * np.power(val,-b_2/2)
            
            grad = cons1 * (-np.log(val) + (b_2*r2*a2) / (b*(b + r2*a2))) * np.power(val,-b_2/2)
            
            grad_beta[ii,jj] = (grad * sum_kernel_mof - f * sum_grad_beta) / (sum_kernel_mof)**2
    
    rz_grad = resize_im(torch.from_numpy(grad_beta), x_shape)
    return  rz_grad

# --- normalisation constant and its gradients
def sum_mof_filter(taille, a, b):
    center = int((taille+1)/ 2)
    X = np.arange(-taille + center, taille - center+1)
    Y = np.arange(-taille + center, taille - center+1)
    kernel_m = np.zeros((taille, taille))
    grad_alpha = np.zeros((taille, taille))
    grad_beta = np.zeros((taille, taille))
    for ii in range(taille):
        for jj in range(taille):
            b_2 = b+2
            a2 = np.power(a,2)
            cons1 = a2 / (4*np.pi)
            r2 = np.power(X[ii],2) + np.power(Y[jj],2)
            val = 1 + (r2*a2) / b
            kernel_m[ii,jj] =  (a2 / (2*np.pi)) * np.power(val,-b_2/2)
            grad_alpha[ii,jj] = (2 - ((b_2 * r2 * a2) / ((b + r2 * a2)))) * np.power(val,-b_2/2) * (a / (2*np.pi))
            grad_beta[ii,jj] = (-np.log(val) + (b_2*r2*a2) / (b*(b + r2*a2))) * np.power(val,-b_2/2) * cons1
    return (np.sum(kernel_m.reshape(-1,1)), np.sum(grad_alpha.reshape(-1,1)), np.sum(grad_beta.reshape(-1,1)))
## --- end Moffat


# --- resize kernel to image's size
def resize_im(kernel, im_size): 

    psf_size = kernel.shape[-1]
    pkernel  = torch.zeros(im_size, device = kernel.device)
    pkernel[:, :, :psf_size, :psf_size] = kernel
    fftkernel = torch.fft.fft2(pkernel)
    
    return fftkernel
 