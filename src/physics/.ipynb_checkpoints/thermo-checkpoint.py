import numpy as np
import scipy.integrate as sin

# Functions to convert Q to Rh and to compute the Bouyancy

levs = np.array([  3.64346569,   7.59481965,  14.35663225,  24.61222   ,
        35.92325002,  43.19375008,  51.67749897,  61.52049825,
        73.75095785,  87.82123029, 103.31712663, 121.54724076,
       142.99403876, 168.22507977, 197.9080867 , 232.82861896,
       273.91081676, 322.24190235, 379.10090387, 445.9925741 ,
       524.68717471, 609.77869481, 691.38943031, 763.40448111,
       820.85836865, 859.53476653, 887.02024892, 912.64454694,
       936.19839847, 957.48547954, 976.32540739, 992.55609512])

P0 = 1e5 # mean



# RELATIVE HUMIDITY RH

# Variables

T0 = 273.16 # in K
T00 = 253.16 # in K

Rv = 461 # in J kg−1 K−1 ---- specific gas constant for water vapor
Rd = 287 # in J kg−1 K−1 ---- specific gas constant for dry air

a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,
                  0.206739458e-7,0.302950461e-5,0.264847430e-3,
                  0.142986287e-1,0.443987641,6.11239921]);
c_liq = -80
a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,
                  0.602588177e-7,0.615021634e-5,0.420895665e-3,
                  0.188439774e-1,0.503160820,6.11147274]);
c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])


# Functions

# saturation vapor pressure with respect to liquid 
def e_liq(T):
    return 100*np.polyval(a_liq,np.maximum(c_liq, T-T0))

# saturation vapor pressure with respect to ice
def e_ice_modified(T):
    return (T>c_ice[0])*e_liq(T)+\
    np.logical_and(T<=c_ice[0], T > T00) *100*np.polyval(a_ice,T-T0)+\
    (T<=T00)*100*np.polyval(a_ice,T00-T0)#* 100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))

def e_ice(T):
    return (T>c_ice[0])*e_liq(T)+\
    np.logical_and(T<=c_ice[0], T > c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])* 100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))

# linear weight for temperatures between T0 and T00
def weight(T):
    return np.maximum( 0, np.minimum( 1, (T-T00)/(T0-T00) ))

# saturation pressure of water vapor
def e_sat_modified(T):
    #return (T>T0)*e_liq(T)+\
    #np.logical_and(T <= T0, T > T00) * (weight(T)*e_liq(T) + (1-weight(T))*e_ice(T))+\
    #(T<=T00)* e_ice(T)
    w = weight(T)
    return w*e_liq(T) + (1-w)*e_ice_modified(T)

def e_sat(T):
    #return (T>T0)*e_liq(T)+\
    #np.logical_and(T <= T0, T > T00) * (weight(T)*e_liq(T) + (1-weight(T))*e_ice(T))+\
    #(T<=T00)* e_ice(T)
    w = weight(T)
    return w*e_liq(T) + (1-w)*e_ice(T)

def e_sat_GG(T):
    Tst = 373.15 
    est = 1013.25
    T0 = 273.16 
    ei0 = 6.1173 

    T_min = T0-20
    
    T_max = T0

    w = (T_max-T)/(T_max-T_min)

    return (T>T_max) * 100* (est * (Tst/T)**5.02808 * 10**(-7.90298*(Tst/T -1)- 1.3816e-7*(10**(11.344*(1-T/Tst))-1)+8.1328e-3*(10**(-3.49149*(Tst/T-1))-1))) +\
    np.logical_and(T<=T_max, T > T_min) * 100 * ( w * (ei0 * ((T0/T)**(-3.56654)) * 10**((-9.09718*(T0/T-1))+0.876793*(1-T/T0))) +\
                                                 (1-w) * (est * (Tst/T)**5.02808 * 10**(-7.90298*(Tst/T -1)- 1.3816e-7*(10**(11.344*(1-T/Tst))-1)+8.1328e-3*(10**(-3.49149*(Tst/T-1))-1)))) +\
    (T<=T_min) * 100 * (ei0 * ((T0/T)**(-3.56654)) * 10**((-9.09718*(T0/T-1))+0.876793*(1-T/T0)))
    

def ConvertToRH(T, p, q):
    return Rv/Rd * (p*q / e_sat_GG(T))


# BUOYANCY B_PLUME

# variables

# values from Toms notebook
g = 9.80616 # 
Lv = 2.501e6 # in J/kg --- latent heat of vaporization of water in standard conditions
cp = 1.00464e3 

# Functions

# saturation specific humidity
def q_sat(T, p):
    return Rd * e_sat_GG(T)/ (Rv * p)


# geopotential Height:
def geo_height(T, p, q):
    return np.concatenate((0*T[:,0:1,:,:] ,
                           - sin.cumulative_trapezoid(x=p , 
                                          y=T/(g*p) * (Rd + (Rv-Rd)*q), 
                                          axis=1)), axis=1)

# the environmental saturated moist static energy in pressure coordinates
def h_sat(T, p, q):
    return Lv*q_sat(T, p) + cp*T + g*geo_height(T, p, q)

def h_par(T, q):
    return Lv*q[:,-1,:,:] + cp*T[:,-1,:,:]  # check shape of the variables


# dimensionless factor
def kappa(T, p):
    return 1 + Lv**2 * q_sat(T, p) / (Rv * cp * T**2)

def ConvertToBuoyancy(T, p, q):
    return g * (h_par(T, q) - h_sat(T, p, q)) / (kappa(T, p) * cp*T)

