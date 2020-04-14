import numpy as np
import matplotlib.pyplot as plt
import qutip
from qutip import *

def w(t, args):
    """calculates and returns the modulated frequency like in "Lit early universe"

    t time at which the frequency is calculated
    args: a list {w0, dwQ, dtQ, dwP, dtP, delay} with necessary arguments
        w0 the unmodulated frequency
        dwQ (strength) and dtQ (duration) of a gaussian shaped quench centered around t=0
        dwP (strength) and dtP (duration) of a parametric modulation of frequency 2 w0 which starts at t = delay
        dtP shoud be an integer multiple of pi/(2 w0) to avoid uncontinuity at t=delay+dtP
        units: all frequencies are circular frequencies with unit MHz, times have unit \mu s"""
    w0 = args[0]
    dwQ = args[1]
    dtQ = args[2]
    dwP = args[3]
    dtP = args[4]
    delay = args[5]

    freq = w0
    # freq += dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2)
    freq += dwQ*np.exp(-0.5*(t/dtQ)**2)
    freq += dwP*np.sin(2*w0*(t-delay))*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1)
    return(freq)

def wdot(t, args):
    """calculates the time derivative of w(t, args) at time t
    check help(w) for further information on args"""
    w0 = args[0]
    dwQ = args[1]
    dtQ = args[2]
    dwP = args[3]
    dtP = args[4]
    delay = args[5]

    # freqD = - dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2)
    freqD = - dwQ*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2)
    freqD += 2*w0*dwP*np.cos(2*w0*t)*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1)
    return(freqD)

# defining the hamiltonian
def H(t, args):
    """calculates the hamiltonian of a harmonic oscillator with modulated frequency
    has an additional term which takes a force proportional to wdot into account

    args (dictonary which carries all arguments except t):
        t time at which the Hamiltonian is calculated (unit \mu s)
        n dimension of the hilbert space (or cutoff dimension for the numerical calculations)
        f0 proportionality constant of the additional force (unit N MHz^2)
        omega(t, omegaArgs) frequency, modulated in time, described by the list of arguments omegaArgs
        omegaDt(t, omegaArgs) time derivative of the frequency
        => in args you need: n, f0, omega, omegaDt, omegaArgs

    This form of imput is necessary to use H in further calculations (mesolve)"""

    f0 = args['f0']
    n = args['n']
    omega = args['omega']
    omegaDt = args['omegaDt']
    omegaArgs = args['omegaArgs']

    ad = create(n)
    a = destroy(n)
    # H0, for the first two terms see Silveri 2017 Quantum_systems_under_frequency_modulation
    ham = omega(t, omegaArgs)*(ad*a+0.5*qeye(n))
    # additional term because of w(t) not constant
    ham += 1j/4*omegaDt(t, omegaArgs)/omega(t, omegaArgs)*(a*a-ad*ad)
    # Force term (9**10^-9 = x0, extent of ground state wave function), see Wittmann diss
    ham += (9*10**-9)/(10**6)*f0*omegaDt(t, omegaArgs)*(ad + a)
    return(ham)


def getParams(psi):
    """calculates for a given state psi:
    alpha: the coherent displacement parameter
    xi: the squeezing parameter
    nBar: the mean photon number
    nT: the photon number due to the thermal excitation DM_t
    returns alpha, xi, nBar, nT

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
    # cos is symmetric to x=0, therefore is the inverse +/- arccos(...)
    # xiT = np.sign(xiT1)*np.arccos(xiT1)
    xiT = np.sign(xiT1)*np.arccos(xiT1)
    xi = xiR*np.exp(1j*xiT)
    # alpha = 0.5*np.sqrt(xV + pV)
    alpha = expect(a, psi)
    # print(alpha)
    nBar = np.abs(expect(num(n), psi))
    # print(nBar)
    # calculates the thermal excitation (assuming DM_psi = D S DM_t S.dag() D.dag())
    psiT = squeeze(n, xi).dag()*displace(n, alpha).dag()*psi*displace(n, alpha)*squeeze(n, xi)
    nT = np.abs(expect(num(n), psiT))

    xic = np.conj(xi)
    psiTc = squeeze(n, xic).dag()*displace(n, alpha).dag()*psi*displace(n, alpha)*squeeze(n, xic)
    nTc = np.abs(expect(num(n), psiTc))

    if nTc < nT:
        return(alpha, xic, nBar, nTc)
    else:
        return(alpha, xi, nBar, nT)


def plotResults(times, result, args):
    """plots the development of the coherent displacement alpha,
    squeezing parameter r, mean excitation number nBar, thermal excitation nT
    together with the time dependant frequency and the force
    arguments:
        times: list of times for which the values should be calculated
        results: list of states (as returned from mesolve) corresponding to times
        args: arguments given to H in the calculation of the dynamics"""
    wList = args['omega'](times, args['omegaArgs'])
    wdotList = args['omegaDt'](times, args['omegaArgs'])

    masterList = [[],[],[],[]]
    for psi in result.states:
        alpha, xi, nBar, nT = getParams(psi)
        masterList[0].append(np.abs(alpha))
        masterList[1].append(np.abs(xi))
        masterList[2].append(nBar)
        masterList[3].append(nT)

    plt.subplot
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    fig.set_size_inches(15.5, 7.5, forward=True)

    ax1.plot(times, masterList[0], label = "np.abs(alpha)")
    ax1.legend()
    ax2.semilogy(times, masterList[1], label = "r")
    ax2.legend()
    ax3.semilogy(times, masterList[2], label = "nBar")
    ax3.semilogy(times, masterList[3], label = "nT")
    ax3.legend()
    ax4.plot(times, wList, label = "w(t)")
    ax4.legend()

    ax5.plot(times, wdotList*args['f0'], label = "F/hbar in N/(Js)")
    ax5.legend()
    plt.show()

    return(0)
