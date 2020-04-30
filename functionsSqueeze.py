import numpy as np
import matplotlib.pyplot as plt
import qutip
import scipy.special as scs
from qutip import *
import time



def wQP(t, args):
    """calculates and returns the modulated frequency like in "Lit early universe"

    t time at which the frequency is calculated
    args: a list {w0, dwQ, dtQ, dwP, dtP, delay} with necessary arguments
        w0 the unmodulated frequency
        dwQ (strength) and dtQ (duration) of a gaussian shaped quench centered around t=0
        dwP (strength) and dtP (duration) of a parametric modulation of frequency 2 w0 which starts at t = delay
        dtP shoud be an integer multiple of pi/(2 w0) to avoid uncontinuity at t=delay+dtP

    units: all frequencies are circular frequencies with unit MHz, times have unit \mu s"""
    w0, dwQ, dtQ, dwP, dtP, delay = args[0], args[1], args[2], args[3], args[4], args[5]

    # freq += dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2)
    freq = w0 + dwQ*np.exp(-0.5*(t/dtQ)**2) # quench
    freq += dwP*np.sin(2*w0*(t-delay))*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1) # parametric
    return(freq)

def wQPdot(t, args):
    """calculates the time derivative of w(t, args) at time t
    check help(wQP) for further information on args"""
    w0, dwQ, dtQ, dwP, dtP, delay = args[0], args[1], args[2], args[3], args[4], args[5]

    # freqD = - dwQ/(np.sqrt(2*np.pi)*dtQ)*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2)
    freqD = - dwQ*np.exp(-0.5*(t/dtQ)**2) * t/(dtQ**2) # quench
    freqD += 2*w0*dwP*np.cos(2*w0*t)*np.heaviside(t-delay,1)*np.heaviside(dtP-(t-delay),1) # parametric
    return(freqD)

def wQQ(t, args):
    """calculates and returns the modulated (two quenches) frequency like in "Lit early universe"

    t time at which the frequency is calculated
    args: a list {w0, dw1, dt1, dw2, dt2, delay} with necessary arguments
        w0 the unmodulated frequency
        dw1/2 (strength) and dt1/2 (duration) of the first/second gaussian shaped quench
        delay: time between the two quenches

    units: all frequencies are circular frequencies with unit MHz, times have unit \mu s"""
    w0, dw1, dt1, dw2, dt2, delay = args[0], args[1], args[2], args[3], args[4], args[5]

    freq = w0
    freq += dw1*np.exp(-0.5*(t/dt1)**2)
    freq += dw2*np.exp(-0.5*((t-delay)/dt2)**2)
    return(freq)

def wQQdot(t, args):
    """calculates the time derivative of wQQ(t, args) at time t
    check help(wQQ) for further information on args"""
    w0, dw1, dt1, dw2, dt2, delay = args[0], args[1], args[2], args[3], args[4], args[5]

    freqD = - dw1*np.exp(-0.5*(t/dt1)**2) * t/(dt1**2)
    freqD += - dw2*np.exp(-0.5*((t-delay)/dt2)**2) * (t-delay)/(dt2**2)
    return(freqD)

# defining the spin phonon coupling hamiltonian
def H_spin_phonon_coupling(w0, wz, Omega, n_LD, n):
    # taken from "time resolved thermalization"
    """returns the Hamiltonian for the spin phonon coupling
    arguments:
        w0: circular frequency of the gap between the two spin levels
        wz: circular frequency of the phonon harmonic oscillator (= trap frequency)
        Omega: rabi frequency
        n_LD: Lamb Dicke parameter
        n: dimension of the phonon hilbert space (or cutoff dimension for the numerical calculations)
    """
    a = destroy(n)
    ad = a.dag()
    sUp = 0.5*(sigmax() + 1j*sigmay())
    sDown = 0.5*(sigmax() - 1j*sigmay())
    C = (1j*n_LD*(ad + a)).expm() - qeye(n)

    H_ps = 0.5*wz*tensor(sigmaz(), qeye(n)) # phonon part
    H_ps += w0*tensor(qeye(2), ad*a) # spin part
    H_ps += 0.5*Omega*(tensor(sUp, C) + tensor(sDown, C.dag())) # coupling
    return(H_ps)

# defining the hamiltonian of the phonon evolution for vaiable w(t)
def H(t, args):
    """calculates the hamiltonian of a harmonic oscillator with modulated frequency
    has an additional term which takes a force proportional to 1/w^2 into account

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
    # with compensation term -f0/w0^2 (e.g. no force in the case of no modulation)
    ham += (9*10**-9)/(10**6)*(f0/(omega(t, omegaArgs)**2) - f0/(omegaArgs[0]**2))*(ad + a)
    # ham += (9*10**-9)/(10**6)*(f0/(omega(t, omegaArgs)**2))*(ad + a)
    return(ham)

def eval_H_QQ(psi, times, args):
    n = args['n']
    ad = create(n)
    a = destroy(n)
    # the following string describes the time dependant frequency
    strWQQ = 'w0 + dw1*exp(-0.5*(t/dt1)**2) + dw2*exp(-0.5*((t-delay)/dt2)**2)'
    # time derivative of the time depandant frequency
    strDWQQ = '- dw1*exp(-0.5*(t/dt1)**2) * t/(dt1**2) + - dw2*exp(-0.5*((t-delay)/dt2)**2) * (t-delay)/(dt2**2)'

    # Hamiltonian in string format, see Silveri 2017 Quantum_systems_under_frequency_modulation
    H = [[ad*a+0.5*qeye(n), strWQQ], [a*a-ad*ad, '1j/4*(' + strDWQQ + ')/(' + strWQQ + ')'], [ad+a, '(9*10**-9)/(10**6)*(f0/(' + strWQQ + ') - f0/(w0**2))']]

    # do the time evolution
    results = mesolve(H, psi, times, args = args)
    return(results)


def RabiTPSR(omega0,  n_LD, n1, n2 = -1):
    """calculates the rabi rate corresponding to |down, n1> -> |up, n2> when doing TPSR
    arguments:
        omega0: rabi frequency of the carrier
        n_LD: Lamb Dicke factor
        n1, n2: motional states, default for n2 is n2 = n1+1
    """
    if n2 == -1:
        n2 = n1+1
    elif n1 > n2: # make sure n1 < n2
        n1, n2 = n2, n1
    dn = n2-n1

    Lpol = scs.genlaguerre(n1, dn)
    fac1 = np.sqrt(scs.factorial(n1)/scs.factorial(n2))
    fac2 = Lpol(n_LD**2)
    return(omega0 * np.exp(-0.5*n_LD**2) * n_LD**dn * fac1 * fac2)



def getParams(psi, calculate_nT = True):
    """calculates for a given state psi:
    alpha: the coherent displacement parameter
    xi: the squeezing parameter
    nBar: the mean photon number
    nT: the photon number due to the thermal excitation DM_t
    calculate_nT: bool, decides if nT will be calculated (takes time), default set to True
        if calculate_nT = False, xi is only correct modulo complex conjugation, nT is set to 0!!!

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
    xiR = np.arcsinh(0.5*np.sqrt(xV + pV - 2 +0j)) # avoid NANs
    if (np.cosh(xiR)*np.sinh(xiR))==0:
        xiT1 = 0
    else:
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
    if calculate_nT:
        psiT = squeeze(n, xi).dag()*displace(n, alpha).dag()*psi*displace(n, alpha)*squeeze(n, xi)
        nT = np.abs(expect(num(n), psiT))

        xic = np.conj(xi)
        psiTc = squeeze(n, xic).dag()*displace(n, alpha).dag()*psi*displace(n, alpha)*squeeze(n, xic)
        nTc = np.abs(expect(num(n), psiTc))

        if nTc < nT:
            return(alpha, xic, nBar, nTc)
        else:
            return(alpha, xi, nBar, nT)
    else:
        return(alpha, xi, nBar, 0)



def plotResults(times, result, args, calculate_nT = True, nSkipp = 1, showProgress = False):
    """plots the development of the coherent displacement alpha,
    squeezing parameter r, mean excitation number nBar, thermal excitation nT
    together with the time dependant frequency and the force
    arguments:
        times: list of times for which the values should be calculated
        results: list of states (as returned from mesolve) corresponding to times
        args: arguments given to H in the calculation of the dynamics
        calculate_nT = True: bool, if nT should be calculated as well (takes time)
        nSkipp = 1: number of states that should be skipped between each plotted point (speeds it up)"""
    t1 = time.time()
    times = times[::nSkipp]
    wList = args['omega'](times, args['omegaArgs'])
    fList = args['f0']/wList**2 - args['f0']/args['omegaArgs'][0]**2

    masterList = [[],[],[],[]]
    nStates = len(result.states[::nSkipp])
    progress = 0
    for psi in result.states[::nSkipp]:
        alpha, xi, nBar, nT = getParams(psi, calculate_nT = calculate_nT)
        masterList[0].append(np.abs(alpha))
        masterList[1].append(np.abs(xi))
        masterList[2].append(nBar)
        masterList[3].append(nT)
        if showProgress:
            progress += 1
            print('\r', "Progress:", round(100*progress/nStates), "%, processing time:", round(time.time() - t1), "s", end = '')

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
    fig.set_size_inches(15.5, 7.5, forward=True)
    ax1.plot(times, masterList[0], label = "np.abs(alpha)")
    ax1.legend()
    ax2.plot(times, masterList[1], label = "r")
    ax2.legend()
    ax3.plot(times, masterList[2], label = "nBar")
    if calculate_nT:
        ax3.plot(times, masterList[3], label = "nT")
    ax3.legend()
    ax4.plot(times, wList, label = "w(t)")
    ax4.legend()
    ax5.plot(times, fList, label = "F/hbar in N/(Js)")
    ax5.legend()
    plt.show()
    return(0)



def scanAlphaXiN(H, psi0, times, args, valueList, whichVal, showProgress = True, skippInLoop = 0):
    """returns quentity of interest (alpha and/or xi and/or nBar) for a given list of valueList

    arguments:
        H: Hamiltonian which governs the time evolution
        psi0: initial states
        times: list of times used to calculate the time evolution
        args: dictionary of arguments given to the hamiltonian (check help(H))

        valueList: list of values for which the quantity should be calculated
        vhichVal: value in args['omegaArgs'] which should be changed according to valueList
        skippInLoop: number of timesteps which should be calculated before the loop over valueList
            this means that these timesteps are calculated only for the value args['omegaArgs'] given in args (not for all values in valueList)
        Not yet implemented: scanA = True, scanX = False, scanN = False
    """
    t1 = time.time()
    alphaList = []
    xiList = []
    nList = []

    # if skippInLoop > 0: calculate the first skippInLoop steps only once (and do the loop only over the rest)
    if skippInLoop > 0:
        times1 = times[:skippInLoop]
        times2 = times[skippInLoop:]
        results = mesolve(H, psi0, times1, args=args)
        psi1 = results.states[-1]
    else:
        times2 = times
        psi1 = psi0

    # calculate time evolution for all values in valueList
    for val in valueList:
        args['omegaArgs'][whichVal] = val # change the value that needs changing
        results = mesolve(H, psi1, times2, args=args) # calculate time evolution

        psi2 = results.states[-1] # final state
        alpha,xi,nBar,_ = getParams(psi2, False) # get alpha
        # alpha = np.sqrt(np.abs(expect(x, psi2)**2) + np.abs(expect(p, psi2)**2)) # get alpha
        alphaList.append(alpha) # save alpha
        xiList.append(xi) # save xi
        nList.append(nBar) # save nBar
        if showProgress:
            print('\r', "Progress: ", round(100*(val-valueList[0])/(valueList[-1]-valueList[0])), "%, processing time:", round(time.time() - t1), "s", end = '')

    return(alphaList, xiList, nList)
