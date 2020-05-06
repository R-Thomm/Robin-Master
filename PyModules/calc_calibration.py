import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import interpolate

import json
def file_read(fn):
    with open(fn, "r") as read_file:
        data_jason = json.load(read_file)
    return data_jason

from cloudpickle import dump, load
def file_write(fn, data):
    with open(fn, 'wb') as f: 
        dump(data, f)

def dBm2P(P_dBm):
    P = (1e-3)*(10.**(P_dBm/10.))
    return P

def V2PdBm(V,R):
    P = (V0**2)/R
    P_dBm = 10.*log10(P/(1e-3)) # dBm
    return P_dBm

def PdBm2V(P_dBm,R):
    P = (1e-3)*(10.**(P_dBm/10.))
    V = np.sqrt(P*R) # Volt
    return V
    
def U_ctl2U_rf(U_ctl):
    R = 1.16 #Ohm
    Z = 50 #Ohm
    Z0 = 213 #Ohm
    # U control -> electrode voltage
    P_dBm = f_UC2dBm(U_ctl)
    P_RF = dBm2P(P_dBm)
    #U_0 = PdBm2V(P_RF,Z)
    U_RF = Z0*np.sqrt(2*P_RF/R)
    #plt.plot(U_ctl,U_RF)
    #plt.xlabel('U ch3p8 in V')
    #plt.ylabel('Voltage @ RF electrodes in V')
    #plt.show()
    return U_RF
    
def mode_freq(x,A,B):
    return np.sqrt(A*x+B)

def inv_mode_freq(x,A,B):
    return (x**2-B)/A

def mode_fit(x,y,func,strt = [0.03,-10.]):
    #mode_freq(x,A,B)
    popt, pcov = curve_fit(func, x, y, p0 = strt, maxfev=50000)
    perr = np.sqrt(np.diag(pcov))
    pfunc = lambda x: func(x,*popt)
    print(popt,perr)
    return popt,perr,pfunc

######## Mixer calibration: U control -> power dBm ########
data_mixer = file_read('../UserData/Calibration/cali_atten_control_mixer.json')
x_mixer = np.array(data_mixer['cv'])
y_mixer = np.array(data_mixer['power'])
func = interpolate.interp1d(x_mixer, y_mixer, kind='quadratic', axis=-1, copy=True, bounds_error=None, assume_sorted=False)
# -30dB attenuator, removed -4dB attenuator compared to measurement 18.12.18 
f_UC2dBm = lambda x : func(x) + 30 + 4


######## Mode frequency calibration: U control -> mode frequency ########
#data_mode = file_read('../UserData/Calibration/cali_mode_freq.json')
#data_mode = file_read('../UserData/Calibration/cali_mode_freq_2019-03-26_15-31-23.json')
data_mode = file_read('../UserData/Calibration/cali_mode_freq_2019-03-27_15-46-42.json')

U_ch3p8     = data_mode['cv']
f_mf        = data_mode['f_mf']
f_hf        = data_mode['f_hf']
#f_mf_gof    = data_mode['f_mf_gof']

if 'pd_avg' in data_mode:
    cv = data_mode['cv']
    pd = data_mode['pd_avg']
else:
    # Load osci trace
    data_trace = file_read('../UserData/Calibration/cali_trace_2019-03-04_17-00-01.json') # T=600us, width=250us, wait=50us, N=1000, 0..9V
    t = np.array(data_trace['t'])*1e6
    data_avg = data_trace['data_avg']

    plt.plot(t,data_avg[0],label='PD AO')
    plt.plot(t,data_avg[1],label='control')
    plt.plot(t,data_avg[2],label='PD refl.')
    plt.plot(t,data_avg[3],label='TTL')
    plt.legend()
    plt.xlabel('Time in us')
    plt.ylabel('Voltage in V')
    plt.show()

    pd = np.array(data_avg[0])
    cv = np.array(data_avg[1])

######## FIT MODE FREQ ########
U_RF = U_ctl2U_rf(U_ch3p8)
popt_mf,perr_mf,func_mf = mode_fit(U_RF,f_mf,mode_freq)
popt_hf,perr_hf,func_hf = mode_fit(U_RF,f_hf,mode_freq)

freq_mf_U_ctrl = lambda x: func_mf(U_ctl2U_rf(x))
freq_hf_U_ctrl = lambda x: func_hf(U_ctl2U_rf(x))

x = np.linspace(0,5,1000)
U = U_ctl2U_rf(x)
f_U_inv = interpolate.interp1d(U, x, kind='quadratic', axis=-1, copy=True, bounds_error=None, assume_sorted=False)

inv_freq_mf_U_ctrl = lambda x: f_U_inv(inv_mode_freq(x,*popt_mf))
inv_freq_hf_U_ctrl = lambda x: f_U_inv(inv_mode_freq(x,*popt_hf))

######## Fit inverse function: p: mode freq. -> PD voltage ########
f_mf_cv = np.array(func_mf(U_ctl2U_rf(cv)))
idx = ~np.isnan(f_mf_cv)
if any(~idx):
    pd = pd[idx]
    f_mf_cv = f_mf_cv[idx]

order = 2
z_pd2mf = np.polyfit(pd,f_mf_cv,order)
p_pd2mf = np.poly1d(z_pd2mf)

z_mf2pd = np.polyfit(f_mf_cv,pd,order)
p_mf2pd = np.poly1d(z_mf2pd)
print('Poly coeff: ', z_mf2pd)

######## PLOT ########
plt.plot(x,freq_mf_U_ctrl(x),label='MF Fit')
plt.plot(x,freq_hf_U_ctrl(x),label='HF Fit')
plt.plot(U_ch3p8,f_mf,'x',label='MF')
plt.plot(U_ch3p8,f_hf,'x',label='HF')
plt.legend()
plt.xlabel('U ch3p8 in V')
plt.ylabel('Mode frequency in MHz')
plt.show()

X_freq = np.linspace(min(f_mf_cv),max(f_mf_cv),1000)
plt.plot(f_mf_cv,pd,'x',label='data')
plt.plot(X_freq,p_mf2pd(X_freq),'-',label='poly fit n=%i \n%s' % (order,str(z_mf2pd)))
plt.xlabel('MF - mode frequency in MHz')
plt.ylabel('Pickup PD voltage in V')
plt.legend()
plt.show()

X_pd = np.linspace(min(pd),max(pd),1000)
plt.plot(pd,f_mf_cv,'x',label='data')
plt.plot(X_pd,p_pd2mf(X_pd),'-',label='poly fit n=%i \n%s' % (order,str(z_pd2mf)))
plt.ylabel('MF - mode frequency in MHz')
plt.xlabel('Pickup PD voltage in V')
plt.legend()
plt.show()

######## WRITE TO FILE ########
data_out = {
        'f_mf':freq_mf_U_ctrl,
        'f_hf':freq_hf_U_ctrl,
        'f_i_mf':inv_freq_mf_U_ctrl,
        'f_i_hf':inv_freq_hf_U_ctrl,
        'f_mf2pd':p_mf2pd,
        'f_pd2mf':p_pd2mf }

from time import localtime, strftime
cali_file = '../UserData/Calibration/cali_func_%s.pkl' % (strftime("%Y-%m-%d_%H-%M-%S", localtime()))
file_write(cali_file, data_out)


