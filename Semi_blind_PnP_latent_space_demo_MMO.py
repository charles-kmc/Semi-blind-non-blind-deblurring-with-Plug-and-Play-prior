
CUDA_LAUNCH_BLOCKING="1"
from collections import defaultdict
import math
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

import time as time
from model.models import *
from utils_SB.utils import *
from utils_SB.kernel import *
from utils_SB.modules import *

import datetime
import os
import sys

import scipy.io

# ---
torch.backends.cudnn.benchmark = True

# Cuda 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def actualconvolvefft(x, H):
    FFTx = torch.fft.fft2(x)
    return torch.real(torch.fft.ifft2(FFTx * H))

### -----

start_time = time.time()
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

with torch.no_grad():
    
    data = os.listdir('images')

    for i in range(len(data)):
        data[i] = 'images/' + data[i]
    
    BSNRdb = 30
    
## --- PHASE: General parameters
   
    max_iter = 50000
    max_iter_wu = 30000
    
    rate = 0.3
    burnin = int(max_iter * rate)
    c_rho = 10000
    c_sigma = 100000
    c_a = 10
    c_b = 10
    absolute = 0
    projection = 1
    for c_ in [100]:
        rho= 218
        # --- Kernel parameters
        taille = 7
        a = 0.4
        b = 0.3
        b_init = 0.1; b_min = 0.1; b_max = 1
        a_init = 0.1; a_min = 0.1; a_max= 1
        snr_min = 15; snr_max = 45
        
        # --- step parameters in the SAPG algorithm
        d_scale = 1
        d_exp = 0.8
    
        # --- To control parameters tuning
        fix_sigma = 0; fix_rho = 0; fix_a = 0; fix_b = 0
        proj = 0
        plots = 1
        if fix_a:
            a_init = a
        if fix_b:
            b_init = b
        
        for alpha in [0.5]:
            for i_im in range(8): 
                
            ## --- PHASE 1: Load true image, blur kernel and blur operator
                filename = data[i_im]
                dataname = filename[7:-4]
                print(f'\n \t Data: {dataname}\n')
                im = cv2.imread(filename, 0) 
                print(f'\n Image shape = {im.shape}\n')
                x1 = torch.Tensor(np.expand_dims(im, axis=(0,1)).astype(np.float32))
                x = x1 / 255.0
                x = x.to(device).detach()
                dimX = torch.numel(x)
                x_shape = x.shape

                # --- Blur operators and Gradients
                h = lambda a,b: Gaussian_filter(taille, a, b).to(device)
                H_FFT = lambda a,b: resize_im(h(a,b), x_shape)
                HC_FFT = lambda a,b: torch.conj(H_FFT(a,b))
                
                diffa_FFTH = lambda a, b: diff_kernel_a(taille, a, b, x_shape).to(device)
                diffb_FFTH = lambda a, b: diff_kernel_b(taille, a, b, x_shape).to(device)
                
                Da_FFTH = lambda x,a,b: actualconvolvefft(x, diffa_FFTH(a,b))
                Db_FFTH = lambda x,a,b: actualconvolvefft(x, diffb_FFTH(a,b))

                A = lambda x, a,b : actualconvolvefft(x, H_FFT(a,b))                 
                AT = lambda x, a,b : actualconvolvefft(x, HC_FFT(a,b))
                

            ## --- PHASE : Precision matrix
                precision_matrix = lambda a,b, sigma, rho: 1 / (torch.abs(H_FFT(a,b))**2 * sigma**2 + rho**2)
                presicion_matrix_x = lambda x,a,b, sigma, rho: torch.real( torch.fft.ifft2( precision_matrix(a, b, sigma, rho) * torch.fft.fft2(x) ) )
                
            ## --- PHASE: Noise and degraded image
                Ax = A(x,a,b)
            
                sigma     = 1 / (torch.linalg.matrix_norm(Ax-torch.mean(Ax), ord='fro')/math.sqrt(torch.numel(x)*10**(BSNRdb/10)))
                s_min = 1 / (torch.linalg.matrix_norm(Ax-torch.mean(Ax), ord='fro')/math.sqrt(torch.numel(x)*10**(snr_min/10)))
                s_max = 1 / (torch.linalg.matrix_norm(Ax-torch.mean(Ax), ord='fro')/math.sqrt(torch.numel(x)*10**(snr_max/10)))
                sigma2 = sigma*sigma
                sigma_sorted = torch.sort(torch.tensor([s_min, s_max]))[0]
                sigma_min = sigma_sorted[0].to(device)
                sigma_max = sigma_sorted[1].to(device)
                
                if fix_sigma:
                    sigma_init = sigma[0,0]
                else:
                    sigma_init = (s_min[0,0] + s_max[0,0]) / 2
                
                rho_min = 50
                rho_max = 600
            
                if fix_rho:
                    rho_init = torch.tensor(rho).to(device)
                else: 
                    rho_init = (rho_min + rho_max) / 2
                
                # --- Observed data y        
                noise = (1 / sigma[0,0]) * torch.randn_like(x).to(device)
                y = Ax + noise
                
            ## ---  PHASE: likelihood, and gradients 
                f = lambda x,a,b, sigma : sigma**2 * (torch.linalg.matrix_norm(y-A(x, a, b), ord='fro')**2.0)/(2.0)
                h_f = lambda x, u, rho : rho**2 * (torch.linalg.matrix_norm(x - u, ord='fro')**2.0)/(2.0)
                grad_h_z = lambda x, u, rho : rho**2 * (u - x)
                
                grad_a = lambda x, a, b, sigma: torch.sum(Da_FFTH(x,a,b) * (A(x,a,b) - y)) * sigma**2
                grad_b = lambda x, a, b, sigma: torch.sum(Db_FFTH(x,a,b) * (A(x,a,b) - y)) * sigma**2
                grad_sigma = lambda x, a, b, sigma: - sigma * (torch.linalg.matrix_norm(y - A(x, a, b), ord='fro')**2.0) + dimX / sigma
                grad_rho   = lambda x, u, rho : - rho * (torch.linalg.matrix_norm(x - u, ord='fro')**2.0) + dimX / rho
                
            ## --- PHASE: PnP parameters setting for warm-up and SAPG
                n_ch, n_lev, ljr = 1, 2.25, 0.002
                L = 1
                L_y =  torch.min( 0.98**2 * sigma_min**2 + rho_min**2, 0.98**2 * sigma_max**2 + rho_max**2 )
                eps = (n_lev / 255.0)**2
                max_lambd = 1.0 / ((2.0*alpha*L)/eps + 4.0*L_y)
                lambd_frac = 0.99
                lambd = max_lambd*lambd_frac

                delta_max = (1.0/3.0)/((alpha*L)/eps+L_y+1/lambd)
                delta_frac = 0.99
                delta = delta_max*delta_frac
                
                # --- Denoiser model
                model = load_model(n_ch, n_lev, ljr, device) 
                
                # --- convex interval for projection
                C_upper_lim = torch.tensor(1.0).to(device)
                C_lower_lim = torch.tensor(0.0).to(device)
                X_wu = y.clone()
                U_wu = y.clone()
                U_denoiser = denoiser(X_wu,model)
                ATy = lambda a, b: AT(y, a, b)
                orig_mse = mse_function(y.to(device).detach(),x).cpu().numpy()
                rmse = []
                vec_mse2 = []; vec_psnr2 = []
                a_it = a_init
                b_it = b_init
                sigma_it = sigma_init
                rho_it = rho_init
                
            ## --- PHASE: Warm-up of the PnP algorithm
                
                if max_iter_wu >0:
                    ATy_wu = ATy(a_init, b_init)
                    for ii in range(max_iter_wu):
                        ## mean of X|Y, Z
                        RHS_wu = ATy_wu * sigma_init**2  + U_wu * rho_init**2
                        X_wu = presicion_matrix_x(RHS_wu, a_init, b_init, sigma_init, rho_init)
                        
                        ## --- Sample from the latent space with PnP-ULA
                        U_wu = U_wu - delta*grad_h_z(X_wu, U_wu, rho_init) + (alpha*delta/eps) * (U_denoiser - U_wu) +\
                                                        (delta/lambd) * (projbox(U_wu,C_lower_lim,C_upper_lim) - U_wu) +\
                                                            math.sqrt(2*delta)*torch.randn_like(U_wu)
                        U_denoiser = denoiser(U_wu, model)

                        # --- Monitoring progression
                        if np.mod(ii, int(max_iter_wu/100)) ==0:
                            print(f'\nIteration = {int(ii / int(max_iter_wu/100))}%')
                        
    
            ## --- PHASE: SAPG Here, we estimate a,b, rho, sigma U and X

                # --- initialisation for the 
                delta_sapg = lambda ii: d_scale * ((ii**(-d_exp)) / dimX)

                rhos = []; sigmas =[]; data_as = []; data_bs = []
                mse_mat = []; psnr_mat = []; psnr_vec = []; mse_vec = []
                psnr_mean_x = []; mse_mean_x = []

                rhos.append(rho_init); sigmas.append(sigma_init.cpu().numpy()); data_as.append(a_init); data_bs.append(b_init)
                U = U_wu.clone() 
                X = X_wu.clone()
                
                ## --- Main loop
                print(f'\n \t Running SAPG algorithm ...\n')
                
                for it in range(1, max_iter):
                    # --- X from X|U,a,b,sigma
                    if max_iter_wu >0:
                        RHS_X = ATy(a_it, b_it) * sigma_it**2  + U * rho_it**2 
                        X = presicion_matrix_x(RHS_X, a_it, b_it, sigma_it, rho_it)
                        X_denoiser = denoiser(X, model)
                        
                        # --- PnP ULA in the latent space
                        U = U - delta*grad_h_z(X, U, rho_it) + (alpha*delta/eps) * (U_denoiser - U) + (delta/lambd) * (projbox(U, C_lower_lim,C_upper_lim) - U) + math.sqrt(2*delta)*torch.randn_like(U)
                        U_denoiser = denoiser(U,model)
                    else:
                        X = x
                        U = x
                        
                    # gradients
                    G_a = grad_a(X,a_it, b_it, sigma_it)
                    G_b = grad_b(X ,a_it, b_it, sigma_it)
                    G_rho = grad_rho(X, U, rho_it)
                    G_sigma = grad_sigma(X, a_it, b_it, sigma_it)
                    
                    # --- Metric in the SAPG stage   
                    temp_mse  = mse_function(X,x)
                    temp_psnr = psnr_function(X,x)
                    psnr_vec.append(temp_psnr)
                    mse_vec.append(temp_mse)
                    
                    if fix_a:
                        a_it = a
                    else:
                        a_temp = torch.tensor(a_it).to(device) - c_a * delta_sapg(it) * G_a
                        a_it = projbox(a_temp, a_min, a_max).cpu().numpy()
                    data_as.append(a_it)
                    
                    # --- update b 
                    if fix_b:
                        b_it = b
                    else:
                        b_temp = torch.tensor(b_it).to(device) - c_b * delta_sapg(it) * G_b
                        b_it = projbox(b_temp, b_min, b_max).cpu().numpy()
                    data_bs.append(b_it)
                    
                    # --- update \rho
                    if fix_rho:
                        rhoii = rho_init
                        rho_it = rho_init
                    else:       
                        rhoii = torch.tensor(rho_it).to(device)  +  c_rho * delta_sapg(it) * G_rho
                        rhoii = rhoii[0,0]
                        rho_it = projbox(rhoii, rho_min, rho_max).cpu().numpy()
                    rhos.append(rho_it)
                    
                    # --- update sigma
                    if fix_sigma:
                        sigmaii = sigma[0,0]
                    else: 
                        sigmaii = torch.tensor(sigma_it).to(device)  +  c_sigma * delta_sapg(it) * G_sigma
                        sigmaii = sigmaii[0,0]
                        
                    sigma_it = projbox(sigmaii ,sigma_min, sigma_max).cpu().numpy()
                    sigmas.append(sigma_it)
                    
                    # --- End Burnin
                    if it == burnin:
                        # Initialise recording of sample summary statistics
                        post_meanvar = welford(X)
                        fouriercoeff = welford(torch.fft.fft2(X).abs())
                    elif it > burnin:
                        # --- update
                        post_meanvar.update(X)
                        fouriercoeff.update(torch.fft.fft2(X))
                        
                        # --- metrics
                        temp_mse2 = mse_function(post_meanvar.get_mean(),x)
                        temp_psnr2 = psnr_function(post_meanvar.get_mean(),x)
                        
                        psnr_mean_x.append(temp_psnr2)
                        mse_mean_x.append(temp_mse2)
                        
                    # --- Monitoring progression within the SAPG stage
                    if np.mod(it, int(max_iter/100)) ==0:
                        print(f'\nIteration = {int(it / int(max_iter/100))}%')
                
                end_time = time.time()
                elapsed = end_time - start_time
                    
                if plots:
                    # --- x true
                    plot1 = plt.figure()
                    plt.imshow(x.detach().cpu().numpy()[0,0], cmap="gray")
                    plt.axis('off')
                    
                    # --- observed data y
                    plot1 = plt.figure()
                    plt.imshow(y.detach().cpu().numpy()[0,0], cmap="gray")
                    plt.axis('off')

                    if max_iter_wu>0:
                        # -- a
                        plot1 = plt.figure()
                        plt.plot(data_as, label = 'a')
                        plt.plot(a * np.ones(len(data_as)), label = '-true-')
                        plt.legend()
                        
                        # --- b
                        plot1 = plt.figure()
                        plt.plot(data_bs, label = 'b')
                        plt.plot(b * np.ones(len(data_bs)), label = '-true-')
                        plt.legend()

                        # --- PSNR mean
                        plot1 = plt.figure()
                        plt.plot(torch.stack(psnr_mean_x).cpu().numpy(), label = '-psnr mean x-')
                        plt.legend()

                        # --- MSE mean
                        plot1 = plt.figure()
                        plt.plot(torch.stack(mse_mean_x).cpu().numpy(), label = '-mse mean x')
                        plt.legend()

                    # --- rho
                    if fix_rho == 0:
                        plot1 = plt.figure()
                        plt.plot(rhos)
                        
                    # --- sigma
                    if fix_sigma == 0:
                        plot1 = plt.figure()
                        plt.plot(sigmas, label = 'sigma')
                        plt.plot(sigma[0,0].cpu().numpy() * np.ones(len(sigmas)), label = f'sigma = {sigma.cpu().numpy()[0,0]:.3f}')
                        plt.legend()
                    
                    # --- post Var X    
                    plot1 = plt.figure()
                    plt.imshow(post_meanvar.get_var().detach().cpu().numpy()[0,0], cmap="gray")
                    plt.axis('off')
                    plt.title('x - posterior variance')
                    
                    # --- post mean X  
                    plot1 = plt.figure()
                    plt.imshow(post_meanvar.get_mean().detach().cpu().numpy()[0,0], cmap="gray")
                    plt.axis('off')
                    plt.title('x - posterior mean')

sys.exit()               
                    