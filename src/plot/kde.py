import numpy as np
from scipy.stats import gaussian_kde
from functools import partial

def my_kde_bandwidth(obj, fac=1./5):
    """
    found on: https://docs.scipy.org/doc/scipy/tutorial/stats/kernel_density_estimation.html
    
    We use Scott's Rule, multiplied by a constant factor. 
    function from the documentation of `gaussian_kde`
    """
    return np.power(obj.n, -1./(obj.d+4)) * fac

def kde_fit(hist, low, up, bw_method = "silverman"):
    y = hist[0]
    i = np.arange(len(hist[1])-1)
    x = (hist[1][i+1]+hist[1][i])/2
    kde = gaussian_kde(x, weights=y/y.sum(), bw_method=bw_method)
    density = kde(np.linspace(low,up,100))
    return density

def get_histogram_kde(data):
    low = np.min(data)
    up = np.max(data)
    
    Xbin = np.arange(low, up, (up-low)/100)
    
    counts, bin_edges = np.histogram(data, bins=Xbin)
    hist = [counts, bin_edges]
    
    tot_number = counts.sum()
    
    fitted = kde_fit(hist, low, up)
    
    bar_x, bar_y = bin_edges[:-1], counts #/(tot_number*(up-low)/100)
    bar_width = np.diff(bin_edges) 
    kde_x, kde_y = np.linspace(low, up, 100), fitted
    
    return bar_x, bar_y, bar_width, kde_x, kde_y