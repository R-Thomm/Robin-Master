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
        n dimension of the hilbert space (or cutoff dimension)
        omega(t, a1, a2, a3, a4, a5, a6) frequency, modulated in time, described by parameters a1, ..., a6
        omegaDt(t, a1, ...) time derivative of the frequency,  described by a1, ..., a6
        => in args you need: n, omega, omegaDt, a1, ..., a6

    This form of imput is necessary to use H in further calculations (mesolve)"""

    omega = args['omega']
    omegaDt = args['omegaDt']
    a1 = args['a1']
    a2 = args['a2']
    a3 = args['a3']
    a4 = args['a4']
    a5 = args['a5']
    a6 = args['a6']
    n = args['n']
    ad = create(n)
    a = destroy(n)
    ham = omega(t, a1, a2, a3, a4, a5, a6)*(ad*a+0.5*qeye(n))
    ham += 1j/4*omegaDt(t, a1, a2, a3, a4, a5, a6)/omega(t, a1, a2, a3, a4, a5, a6)*(a*a-ad*ad)
    return(ham)

def getParams(psi):
    """calculates for a given state psi:
    alpha: the coherent displacement parameter
    xi: the squeezing parameter
    nBar: the mean photon number
    nT: the photon number of the thermal state DM_t
    returns the following list: {alpha, xi, nBar, nT}

    assumes that psi can be written as DM_psi = D(alpha) S(xi) DM_t S(xi).dag() D(alpha).dag()
    further assumes that the thermal excitation is close to the vacuum"""
    n = psi.dims[0][0]
    ad = create(n)
    a = destroy(n)
    x = (ad + a)
    p = 1j*(ad - a)

    xV = variance(x, psi)
    pV = variance(p, psi)

    # calculated by hand, assuming t = 0 (e.g. DM_t = |0><0|)
    xiR = np.arcsinh(0.5*np.sqrt(xV + pV - 2 +0j))
    xiT1 = 0.25*(pV - xV)/(np.cosh(xiR)*np.sinh(xiR))
    xiT = -np.sign(xiT1)*np.arccos(xiT1)
    xi = xiR*np.exp(1j*xiT)
    # alpha = 0.5*np.sqrt(xV + pV)
    alpha = expect(a, psi)
    # print(alpha)
    nCS = np.abs(alpha)**2 + np.sinh(xiR)**2
    # print(nCS)
    nBar = expect(ad*a, psi)
    # print(nBar)
    nT = np.abs(nBar - nCS)
    # print(nT)

    return(alpha, xi, nBar, nCS, nT)
