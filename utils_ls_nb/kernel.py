
import torch
import torch.nn as nn
import numpy as np
import math
from scipy import signal
from statsmodels.tsa.stattools import acf as autocorr

# Cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- resize kernel to image's size ---
def resize_im0(kernel, im_size): 

    psf_size = kernel.shape[-1]
    pkernel  = torch.zeros(im_size, device = kernel.device)
    if pkernel.ndim == 2:
        pkernel[ :psf_size, :psf_size] = kernel
    elif pkernel.ndim == 3:
        pkernel[ :, :psf_size, :psf_size] = kernel
    elif pkernel.ndim == 4:
        pkernel[:, :, :psf_size, :psf_size] = kernel
    else:
        raise ValueError 
    
    fftkernel = torch.fft.fft2(pkernel)
    
    return fftkernel



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

def diff_kernel_b(taille, a,b, x_shape):
    
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
    
## --- Moffat filter
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

# --- resize kernel to image's size
def resize_im0(kernel, im_size): 

    psf_size = kernel.shape[-1]
    pkernel  = torch.zeros(im_size, device = device)
    if pkernel.ndim == 2:
        pkernel[ :psf_size, :psf_size] = kernel
    elif pkernel.ndim == 3:
        pkernel[ :, :psf_size, :psf_size] = kernel
    elif pkernel.ndim == 4:
        pkernel[:, :, :psf_size, :psf_size] = kernel
    else:
        raise ValueError 
    
    fftkernel = torch.fft.fft2(pkernel)
    
    return fftkernel
# psf2otf copied/modified from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
def resize_im(psf, shape=None):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if torch.is_tensor(psf):
        device = psf.device
        psf = psf.cpu().numpy()
    else:
        pass
    
    if type(shape) == type(None):
        shape = psf.shape
    shape = np.array(shape)
    if np.all(psf == 0):
        # return np.zeros_like(psf)
        return np.zeros(shape)
    if len(psf.shape) == 1:
        psf = psf.reshape((1, psf.shape[0]))
    inshape = psf.shape
    
    if len(shape) == 2:
        psf = zero_pad(psf, shape, position='corner')
    elif len(shape) == 3:
        psf = zero_pad(psf, shape[1:], position='corner')
    elif len(shape) == 4:
        psf = zero_pad(psf, shape[2:], position='corner')
    else:
        raise ValueError
        
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    # Compute the OTF
    otf = np.fft.fft2(psf, axes=(0, 1))
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return torch.from_numpy(otf).to(device)

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    if len(shape) == 3:
        dshape = shape[1:] - imshape
        pad_img = np.zeros(shape[1:], dtype=image.dtype)
    if len(shape) == 4:
        dshape = shape[2:] - imshape
        pad_img = np.zeros(shape[2:], dtype=image.dtype)
    if len(shape) <3:
        dshape = shape - imshape
        pad_img = np.zeros(shape, dtype=image.dtype)
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


