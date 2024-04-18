
#%% ## Import pakacges
import numpy as np
import matplotlib.pyplot as plt
import torch

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf as autocorr
import arviz


def plot_ACF(traces, Lags, Path, img_name, pic_name):
    fig = plt.figure()
    autocorr_plot1,ax1=plt.subplots(figsize=(6,5))
    ax1.set_xlabel('Lag')
    if len(traces)>1:
        for trace in traces:
            ess = ESS(trace)
            plot_acf(trace, lags=Lags, ax=ax1,label = f"ESS = {int(np.round(ess))}")
    else:
        ess = ESS(traces[0])
        plot_acf(traces[0], lags=Lags, ax=ax1,label = f"ESS = {int(np.round(ess))}")
               
    handles, labels= ax1.get_legend_handles_labels()
    handles=handles[1::2]
    labels =labels[1::2]
    ax1.set_ylim([-1,1.3])
    ax1.set_title('ACF' , fontsize = 10)
    ax1.legend(handles=handles, labels=labels,loc='best',shadow=True, numpoints=2)
    autocorr_plot1.savefig(Path+"/"+img_name+"_real_ACF_" + pic_name+ ".png")

def ACF(fftXchain_pnp):
    
    real_fftXchain_pnp = torch.abs(torch.tensor(fftXchain_pnp))
    del fftXchain_pnp
    
    # --- trace variance of the chain
    var_real_fftXchain_pnp = torch.var(real_fftXchain_pnp, dim = 0)
    
    # --- lower and faster traces of the chain
    data_real_pnp_max = real_fftXchain_pnp[:,torch.argmax(var_real_fftXchain_pnp)]
    data_real_pnp_min = real_fftXchain_pnp[:,torch.argmin(var_real_fftXchain_pnp)]
    data_real_pnp_median = real_fftXchain_pnp[:,list(var_real_fftXchain_pnp).index(torch.median(var_real_fftXchain_pnp.reshape(-1)))]

    return data_real_pnp_min.numpy(), data_real_pnp_median.numpy(), data_real_pnp_max.numpy()
    
def ESS(arr):
    ess = arviz.ess(arr)
    return ess
