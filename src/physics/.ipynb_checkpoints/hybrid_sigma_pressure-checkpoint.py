import numpy as np

# from metpy.interpolate import log_interpolate_1d
# from scipy.interpolate import interp1d

def get_p(PS, hyam, hybm, P0 = 1e5, N_time = 7320, N_lat = 29, N_lon = 47, N_lev = 32):
    # data arrays!
    hyam = hyam.expand_dims(dim={"time" : N_time, "lat": N_lat, "lon" : N_lon}, axis=(0,2,3))
    hybm = hybm.expand_dims(dim={"time" : N_time, "lat": N_lat, "lon" : N_lon}, axis=(0,2,3))
    PS = PS.expand_dims(dim={"lev" : N_lev}, axis=1)
    return hyam * P0 + hybm * PS



def To_pCoord(var, p, p_levs):
    'THIS DOES NOT WORK: SEE INTERPOLATION EXAMPLE IN NOTEBOOK "Datasets_for_ML" '
    """
    p = p.stack(flat_dim=[p.dims[0], p.dims[2],p.dims[3]]) 
    var = var.stack(flat_dim=[var.dims[0], var.dims[2],var.dims[3]]) 
    return interp1d(
        np.log(p), var, axis=0, kind="linear", fill_value="extrapolate", bounds_error=False
    )(np.log(p_levs))
    # return log_interpolate_1d(
    #     p_levs,  # Desired pressure levels (1D)
    #     p,         
    #     var,      
    #     axis=1 
    # )"""



