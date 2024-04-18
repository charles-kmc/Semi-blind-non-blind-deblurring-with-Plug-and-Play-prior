from collections import defaultdict
import math
import torch
import kornia
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms as T

from utils_ls_nb.utils import *
from utils_ls_nb.kernel import *
from utils_ls_nb.acf import *
from utils_ls_nb.modules import *
import time as time
from tqdm.auto import tqdm

import datetime
import os
import random 
import sys
import scipy.io


#
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()
with torch.no_grad():
    
    data = os.listdir('images')

    for i in range(len(data)):
        data[i] = 'images/' + data[i]
 
    # PHASE: General parameters

    max_iter_wu = 50000
    max_iter = 300000
    chain_size = 1500
    rate = 0.5
    Ranges = int(max_iter*(1 - rate) / chain_size)
    
    burnin = int(max_iter * rate)
    plots = 1
    Lags = 100 
    save_data = 1
        
    ######
    
    c_rho = 100
    c_sigma = 1000000
    d_scale = 1
    d_exp = 0.8
    fix_sigma = 0
    fix_rho = 0
    rotation = True

    rho= 218
    ANGLE = [0,1,2,3]
    alpha = 1.5
    for i_im in range(1):
        filename = data[i_im]
        dataname = filename[7:-4]
        print(f'\n \t Data: {dataname}\n')
        print(f'\t +++++ c_sigma = {c_sigma} \t c_rho = {c_rho}\n\n')
    
    # PHASE 1: Load true image, blur kernel and blur operator
        im = cv2.imread(filename, 0) 
        print(f'\n Image shape = {im.shape}\n')
        x1 = torch.Tensor(np.expand_dims(im, axis=(0,1)).astype(np.float32))
        x = x1 / 255.0
        x = x.to(device).detach()
        dimX = torch.numel(x)
    
        h = Gaussian_filter(7, 0.4, 0.3)
        h = h.to(device)
        H_FFT = resize_im(h, x.shape)

        A = lambda x : actualconvolve(x, torch.unsqueeze(h,0))                   
        AT = lambda x : actualconvolve(x, torch.unsqueeze(h,0), flipkern = True) 
        
    # PHASE : Precision matrix
        precision_matrix = lambda sigma, rho: 1 / (torch.abs(H_FFT)**2 * sigma**2 + rho**2)
        presicion_matrix_x = lambda x, sigma, rho: torch.real( torch.fft.ifft2( \
                                            precision_matrix(sigma, rho) * torch.fft.fft2(x) 
                                            ) )
        
    # PHASE: Noise and degraded image
        BSNRdb = 30
        snr_min = 15
        snr_max = 45
        sigma     = 1 / (torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')\
                                                    /math.sqrt(torch.numel(x)*10**(BSNRdb/10)))
        s_min = 1 / (torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')\
                                                    /math.sqrt(torch.numel(x)*10**(snr_min/10)))
        s_max = 1 / (torch.linalg.matrix_norm(A(x)-torch.mean(A(x)), ord='fro')\
                                                    /math.sqrt(torch.numel(x)*10**(snr_max/10)))
        sigma2 = sigma*sigma
        sigma_sorted = torch.sort(torch.tensor([s_min, s_max]))[0]
        sigma_min = sigma_sorted[0].to(device)
        sigma_max = sigma_sorted[1].to(device)
        
        # --- sigma initialisation ---
        if fix_sigma:
            sigma_init = sigma.squeeze()
        else:
            sigma_init = (s_min.squeeze() + s_max.squeeze()) / 2
        
        # --- Rho initialisation ---
        rho_min = torch.tensor(50, device = device)
        rho_max = torch.tensor(800, device = device)
    
        if fix_rho:
            rho_init = torch.tensor(rho).to(device)
        else: 
            rho_init = (rho_min + rho_max) / 2
        
        # --- Observed image y ---
        Ax = A(x)
        noise = (1 / sigma.squeeze()) * torch.randn_like(x)
        y = Ax + noise

    # PHASE: likelihood, and gradients 
        f = lambda x, sigma : sigma * (torch.linalg.matrix_norm(y-A(x), ord='fro')**2.0)/(2.0)
        h_ls = lambda x, u, rho : rho * (torch.linalg.matrix_norm(x-u, ord='fro')**2.0)/(2.0)
        grad_h_ls_z = lambda x, u, rho : rho**2 * (u-x)
        
        grad_sigma = lambda x, sigma : - sigma * (torch.linalg.matrix_norm(y-A(x), ord='fro')**2.0) + dimX / sigma
        grad_rho   = lambda x, u, rho : - rho * (torch.linalg.matrix_norm(x-u, ord='fro')**2.0) + dimX / rho

    # PHASE: PnP parameters setting for warm-up and SAPG
        L = 1.0
        n_lev = 5
        Lip = "Lipschitz_max_" 
        L_y =  torch.max( sigma_min**2, sigma_max**2) * 0.98**2        
        eps = (n_lev / 255.0)**2
        max_lambd = 1.0/((2.0*alpha*L)/eps + 4.0*L_y)
        lambd_frac = 0.99
        lambd = max_lambd*lambd_frac

        delta_max = (1.0/3.0)/((alpha*L)/eps+L_y + 1/lambd)
        delta_frac = 0.99
        delta = delta_max*delta_frac
        
        normalise = True
        model_type = "RealSN_DnCNN"
        model = load_model_ryu(model_type, n_lev, device) # Load model weights
        C_upper_lim = torch.tensor(1.0, device = device)
        C_lower_lim = torch.tensor(0.0, device = device)

        PosteriorMeanChain_fft = []
        datas = defaultdict()

        # Initialisation
        X_wu = y.clone()
        U_wu = y.clone()
        r_val = random.choice(ANGLE)
        U_denoiser = denoiser_ryu(X_wu, 
                                  model, 
                                  eps, 
                                  r_val, 
                                  rotation = rotation,
                                  normalise = normalise
                                )
        ATy = AT(y)

        mse = torch.nn.MSELoss()
        orig_mse = mse(y.to(device).detach(),x).cpu().numpy()
        rmse = []
        vec_mse2 = []; vec_psnr2 = []
        metric = Metrices()
        
    # PHASE: Warm-up of the PnP algorithm
        if max_iter_wu >0:
            for ii in range(max_iter_wu):
                
                ## mean of X|Y, Z
                RHS_wu = ATy * sigma_init**2  + U_wu * rho_init**2
                X_wu = presicion_matrix_x(RHS_wu, 
                                          sigma_init, 
                                          rho_init
                                        )
                X_wu = projbox(X_wu,C_lower_lim,C_upper_lim)
                
                ## --- Sample from the latent space with PnP-ULA
                U_wu = U_wu - delta*grad_h_ls_z(X_wu, U_wu, rho_init) +\
                                (alpha*delta/eps) * (U_denoiser-U_wu) + \
                                    (delta/lambd) * (projbox(U_wu,C_lower_lim,C_upper_lim)-U_wu) + \
                                        math.sqrt(2*delta)*torch.randn_like(U_wu)

                r_val = random.choice(ANGLE)
                U_denoiser = denoiser_ryu(U_wu, 
                                          model, 
                                          eps, 
                                          r_val, 
                                          rotation = rotation,
                                          normalise = normalise 
                                        )
            
                if np.mod(ii, int(max_iter_wu/100)) ==0:
                    print(f'\nIteration = {int(ii / int(max_iter_wu/100))}%')
                
            
            
    # PHASE: SAPG Algorithm
        delta_sapg = lambda ii: d_scale * ((ii**(-d_exp)) / dimX)
        
        ## -- INITIALISATIONS
        rhos = []; sigmas =[]
        mean_rhos = []; mean_sigmas = []
        psnr_vec = []; mse_vec = []
        psnr_mean_x = []; mse_mean_x = []; psnr_mean_x_denoiser = []
        rhos.append(rho_init); sigmas.append(sigma_init)
        
        U = U_wu 
        X = X_wu
        
        ## --- Main loop
        print(f'\n \t c_sigma = {c_sigma} \t c_roh = {c_rho}\n')
        
        for it in range(1, max_iter):
            
            # --- image of interest
            RHS_X = ATy * sigmas[it-1]**2  + U * rhos[it-1]**2 
            X = presicion_matrix_x(RHS_X, sigmas[it-1], rhos[it-1])
            r_val = random.choice(ANGLE)
            X_denoiser = denoiser_ryu(X,
                                      model,
                                      eps, 
                                      r_val, 
                                      rotation = rotation,
                                      normalise = normalise
                                    )
            X = projbox(X,C_lower_lim,C_upper_lim)
            
            # --- latent variable
            U = U - delta*grad_h_ls_z(X, U, rhos[it-1]) + (alpha*delta/eps) * (U_denoiser - U) \
                                    + (delta/lambd) * (projbox(U, C_lower_lim,C_upper_lim) - U) \
                                        + math.sqrt(2*delta)*torch.randn_like(U)
            r_val = random.choice(ANGLE)
            U_denoiser = denoiser_ryu(U, 
                                      model, 
                                      eps, 
                                      r_val, 
                                      rotation = rotation,
                                      normalise = normalise
                                    )
                
            
            # --- metrics ---
            temp_mse  = metric.mse_function(X,x)
            temp_psnr = metric.psnr_function(X,x)
            psnr_vec.append(temp_psnr)
            mse_vec.append(temp_mse)
                
            # --- rho updated ---
            G_rho = grad_rho(X, U, rhos[it-1])
            if fix_rho:
                rhoii = rho_init
                rho_it = rho_init
            else:       
                rhoii = rhos[it-1] +  c_rho * delta_sapg(it) * G_rho
                rhoii = rhoii.squeeze()
                rho_it = projbox(rhoii, rho_min, rho_max)
            rhos.append(rho_it)
            
            # --- sigma updated ---
            G_sigma = grad_sigma(X, sigmas[it-1])
            if fix_sigma:
                sigmaii = sigma.squeeze()
            else: 
                sigmaii = sigmas[it-1] +  c_sigma * delta_sapg(it) * G_sigma
                sigmaii = sigmaii.squeeze()
                
            sigma_it = projbox(sigmaii ,sigma_min, sigma_max)
            sigmas.append(sigma_it)
            
            if it == burnin:
                # Initialise recording of sample summary statistics
                mean_sigmas = welford(sigma_it)
                mean_rhos = welford(rho_it)
                post_meanvar = welford(X)

            elif it > burnin:
                # -- mse
                temp_mse2 = metric.mse_function(post_meanvar.get_mean(),x)
                # -- psnr
                temp_psnr2 = metric.psnr_function(post_meanvar.get_mean(),x)
                # -- update
                post_meanvar.update(X)
                # -- append value
                psnr_mean_x.append(temp_psnr2)
                mse_mean_x.append(temp_mse2)
                
            if np.mod(it, int(max_iter/100)) ==0:
                print(f'\nIteration = {int(it / int(max_iter/100))}%')

        end_time = time.time()
        elapsed = end_time - start_time
        print(f'\nElapsed time = {elapsed}\n')
        
        if plots:
            # --- true image
            plot1 = plt.figure()
            plt.imshow(x.detach().cpu().numpy()[0,0], cmap="gray")
            plt.axis('off')
            plt.title('x')

            plot1 = plt.figure()
            plt.imshow(y.detach().cpu().numpy()[0,0], cmap="gray")
            plt.axis('off')
            plt.title('y')

            if max_iter_wu>0:
                plot1 = plt.figure()
                plt.plot(torch.stack(psnr_mean_x).cpu().numpy(), label = '-psnr mean x-')
                plt.title('psnr')
                plt.legend()
                
                plot1 = plt.figure()
                plt.plot(torch.stack(mse_mean_x).cpu().numpy(), label = '-mse mean x')
                plt.title('mse')
                plt.legend()

            if fix_rho == 0:
                plot1 = plt.figure()
                plt.plot(torch.stack(rhos).cpu().numpy())
                plt.title('rho')

            if fix_sigma == 0:
                plot1 = plt.figure()
                plt.plot(torch.stack(sigmas).cpu().numpy(), label = 'sigma')
                plt.plot(sigma[0,0].cpu().numpy() * np.ones(len(sigmas)), label = f'sigma = {sigma.cpu().numpy()[0,0]:.3f}')
                plt.legend()
                plt.title('Sigma')
                
            plot1 = plt.figure()
            plt.imshow(post_meanvar.get_var().detach().cpu().numpy()[0,0], cmap="gray")
            plt.axis('off')
            plt.title('x - posterior variance')

            plot1 = plt.figure()
            plt.imshow(post_meanvar.get_mean().detach().cpu().numpy()[0,0], cmap="gray")
            plt.axis('off')
            plt.title('x - posterior mean')
sys.exit()