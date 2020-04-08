import numpy as np
import qutip
from qutip import *

def w(t, w0, dwQ =0, dtQ =1, dwP =0, dtP =0, delay =0):
    """calculates and returns the modulated frequency

    args:
        w0 the unmodulated frequency, t time at which the frequency is calculated (t and w0 are mandatory)
        dwQ (strength) and dtQ (duration) of a gaussian shaped quench centered around t=0
        dwP (strength) and dtP (duration) of a parametric modulation of frequency 2 w0 which starts at t = delay
        dtP shoud be an integer multiple of pi/(2 w0) to avoid uncontinuity at t=delay+dtP

    default parameters describe an unmodulated signal (e.g. return w0 for all t)"""
    freq = w0
    freq += dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2)
    freq += dwP*np.sin(2*w0*(t-delay))*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1)
    return(freq)

def wdot(t, w0, dwQ =0, dtQ =1, dwP =0, dtP =0, delay =0):
    """calculates the time derivative of w(t, w0, ...), check help(w) for further information on the parameters"""
    freqD = - dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2)
    freqD += 2*w0*dwP*np.cos(2*w0*t)*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1)
    return(freqD)

# defining the hamiltonian

def H(t, args):
    """calculates the hamiltonian of a harmonic oscillator with modulated frequency

    args (dictonary which carries all parameters (all are mandatory) except t):
        t time at which the Hamiltonian is calculated
        w0 the unmodulated frequency
        dwQ (strength) and dtQ (duration) of a gaussian shaped quench centered around t=0
        dwP (strength) and dtP (duration) of a parametric modulation of frequency 2 w0 which starts at t = delay
        dtP shoud be an integer multiple of pi/(2 w0) to avoid uncontinuity at t=delay+dtP

    This form of imput is necessary to use H in further calculations (mesolve)"""

    w0 = args['w0']
    dwQ = args['dwQ']
    dtQ = args['dtQ']
    dwP = args['dwP']
    dtP = args['dtP']
    delay = args['delay']
    n = args['n']
    ad = create(n)
    a = destroy(n)
    ham = w(t, w0, dwQ, dtQ, dwP, dtP, delay)*(ad*a+0.5*qeye(n))
    ham += 1j/4*wdot(t, w0, dwQ, dtQ, dwP, dtP, delay)/w(t, w0, dwQ, dtQ, dwP, dtP, delay)*(a*a-ad*ad)
    return(ham)
