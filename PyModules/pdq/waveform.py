import numpy as np

from scipy.special import erfc
from scipy.optimize import curve_fit
import inspect

def func_zeros(n):
   return [0.]*n

def linear_ramp(t,m,b):
    return m*t+b

def gauss(t,t0,sigma):
    x = (t-t0)/sigma
    return np.exp(-.5*x**2)

def d_gauss(t,t0,sigma):
    x = (t-t0)/sigma
    return -x*np.exp(-.5*x**2)

def b_ex_1(t, A, frac, t_ex, t0):
    return A*(1-frac*(np.tanh(1./t_ex**2*(t-t0))+1.)/2.)
   
def b_ex_2(t, A, frac, t_ex, t0):
    return A-frac*gauss(t,t0,t_ex)

def d_b_ex_2(t, A, frac, t_ex, t0):
    dgauss = d_gauss(t,t0,t_ex)
    dy_max = np.max(dgauss)
    dy = np.abs(dgauss)/dy_max
    return A-frac*dy
    #return A-frac*abs(d_gauss(t,t0,t_ex))
    
def ramp(N = 500, T = 20., y = [0.,0.,1.,0.,0.]):
    #y = [0.]*10+[0.,1.,1.,.5,.5,1.,1.,0.]+[0.]*10
    t = np.arange(len(y))*(T/(len(y)-1))
    T = np.linspace(0.,T,num=N,endpoint=True)
    Y = np.interp(T,t,y)
    return T,Y

def step_zero_one(s):
    a = np.array(s>0,dtype=float)*s
    b = np.array([1]*len(a))
    m = np.min([a,b],axis=0)
    return m

def smooth_step(x):
    s = step_zero_one(x)
    return (126+(-420+(540+(-315+70*s)*s)*s)*s)*(s**5)
    
def smooth_step_eu(t, t0, dt):
    #x = (t-t0+dt/2.)/dt #center
    x = (t-t0)/dt
    x = step_zero_one(x)
    #return -20*(x**7)+70*(x**6)-84*(x**5)+35*(x**4)
    return ((((-20*x)+70)*x-84)*x+35)*(x**4)
    
def smooth_step_to(t,t0,A_low,A_high):
    x = -(t-t0)
    y = smooth_step(x)
    y = y*(A_high-A_low) + A_low
    return y

def smooth_dip(t,t1,t2,a1,a2):
    t_r = (t-t2)
    t_l = (t1-t)
    y = smooth_step(t_l) + smooth_step(t_r)
    y = y*(a1-a2) + a2
    return y
    
def waveform(t):
    T = t[-1]
    t_low_edge_l = (4./10.)*T
    t_low_edge_r = (6./10.)*T
    t_l = (1./10.)*T
    t_r = (9./10.)*T
    A_high = 1.
    A_low = .5
    dip = smooth_dip(t,t_low_edge_l,t_low_edge_r,A_high,A_low)
    low_l = smooth_step(t-t_l)
    low_r = smooth_step(-t+t_r)
    control = dip*low_l*low_r
    return control
    
def waveform_dip(t,p_l=1./3.,p_r=2./3.):
    T = t[-1]
    t1 = p_l*T
    t2 = p_r*T
    A_high = 1.
    A_low = .5
    y1 = smooth_step_to(t,t1,A_low,A_high)
    y2 = smooth_step_to(-t,-t2,A_low,A_high)
    y = 2.*y1*y2
    return y

def erfc_step(x,x0,s,A,C):
    return (A/2.)*erfc(-s*(x-x0))+C

def resample(t_new,t,x):
    y = np.interp(t_new, t, x)
    return y

def fit_curve(func,t,y,p0,bounds):
    popt, pcov = curve_fit(func, t, y, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    var_names = inspect.getargspec(func).args[1:]
    report = 'fit: '
    for i,(val,err,arg) in enumerate(zip(popt,perr,var_names)):
        report += ('%s = %.2e (%.2e), '%(arg,val,err))
    print(report)
    return popt,perr

