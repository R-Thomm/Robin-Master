import numpy as np
import matplotlib.pyplot as plt
import qutip
import multiprocessing as mp
import scipy.special as spe
from qutip import *
import time


def RabiTPSR(omega0,  n_LD, n1, n2):
    """calculates the rabi rate corresponding to |down, n1> -> |up, n2> when doing TPSR
    arguments:
        omega0: rabi frequency of the carrier
        n_LD: Lamb Dicke factor
        n1, n2: motional states
    """

    if n1 > n2: # make sure n1 < n2
        n1, n2 = n2, n1

    if n1 < 0:
        return(0)

    if n2==n1+1:
        return(omega0 * np.exp(-0.5*n_LD**2) * n_LD * np.sqrt(1/n2) * spe.eval_genlaguerre(n1, 1, n_LD**2))
    elif n2==n1:
        return(omega0 * np.exp(-0.5*n_LD**2) * spe.eval_genlaguerre(n1, 0, n_LD**2))
    else:
        dn = np.abs(n2-n1)
        return(omega0 * np.exp(-0.5*n_LD**2) * n_LD**dn * np.sqrt(spe.factorial(n1)/spe.factorial(n2)) * spe.eval_genlaguerre(n1, dn, n_LD**2))


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

    H_sp = 0.5*wz*tensor(sigmaz(), qeye(n)) # phonon part
    H_sp += w0*tensor(qeye(2), ad*a) # spin part
    H_sp += 0.5*Omega*(tensor(sUp, C) + tensor(sDown, C.dag())) # coupling
    return(H_sp)

# there is an updated version in the squeezed notebook!!!
def eval_H_spin_phonon_coupling(psi, times, args, options=0, expect=None, n_LD_small = False, crb=1):
    """evaluates the time evolution of the state psi of internal spin coupled with a harmonic oscillator
    the hamiltonian has an additional term which takes a force proportional to 1/w^2 into account

    parameters:
        psi: initial state for the time evolution (should have dimension n, see below)
            of the form tensor(state1, state2), state1 spin state (2-dim), state2 state harm. osc. (n-dim)
        times: list of times for which the state should be calculated
        args: a dictionary with the following entries:
            n: dimension of the phonon hilbert space (or cutoff dimension for the numerical calculations)
            n_LD: Lamb Dicke parameter
            w0: spin frequency
            wz: harm. osc. frequency
            Omega: Rabi frequency (= coupling strength)
        options: possible options for the solver (mesolve)
        expect: operator, for which the expectation value should be calculated for all t in times
        n_LD_small: assumes n_LD is close to zero (so that C can be calculated without exp)

    returns:
        a list of states (evolution of psi, one for each t in times)"""
    n = args['n']
    n_LD = args['n_LD']
    w0 = args['w0']
    wz = args['wz']
    Omega = args['Omega']

    a = destroy(n)
    ad = a.dag()
    sUp = 0.5*(sigmax() + 1j*sigmay())
    sDown = 0.5*(sigmax() - 1j*sigmay())
    if n_LD_small:
        C = 1j*n_LD*(ad + a)
    else:
        C = (1j*n_LD*(ad + a)).expm() - qeye(n)

    H_sp = [[tensor(sigmaz(), qeye(n)), '0.5*wz']]
    H_sp.append([tensor(qeye(2), ad*a), 'w0'])
    H_sp.append([tensor(sUp, C) + tensor(sDown, C.dag()), '0.5*Omega'])

    if options==0:
        results = mesolve(H_sp, psi, times, None, expect, args = args)
    else:
        results = mesolve(H_sp, psi, times, None, expect, args = args, options=options)

    return(results)

def T_l_n(t, l, n, d, Omega, n_LD):
    """calculates and returns the matrix from equation (84) in LBM+03
    parameters:
        t: timepoint
        l: number of sideband (+1 first blue, -1 first red, 0 carrier, +2 second blue, ...)
        n: number of motional state (e.g. consider transition |g, n> <-> |e, n+l>)
        d: detuning of the laser from sideband
        Omega: Rabi frequency
        n_LD: Lamb Dicke parameter
    """

    # initialize Matrix
    mat = np.full((2,2), 0+0j)

    # make the following calculation cleaner
    Orabi = RabiTPSR(Omega, n_LD, n, n+l)
    f = np.sqrt(d**2 + Orabi**2)

    if f == 0 and d==0: # avoid dividung by zero (calculated using Hopital) and speed calculation up
        mat = np.identity(2)
    elif d == 0: # speed calculation up
        mat[(0,0)] = np.cos(0.5*f*t)
        mat[(0,1)] = -1j*Orabi/f*np.exp(1j*(np.abs(l)*np.pi*0.5))*np.sin(0.5*f*t)
        mat[(1,0)] = -1j*Orabi/f*np.exp(-1j*np.abs(l)*np.pi*0.5)*np.sin(0.5*f*t)
        mat[(1,1)] = np.cos(0.5*f*t)

    elif f == 0: # avoid dividung by zero (calculated using Hopital)
        mat[(0,0)] = np.exp(-1j*0.5*d*t) + 1j*d*0.5*t
        mat[(0,1)] = -1j*Orabi*np.exp(1j*(np.abs(l)*np.pi*0.5-0.5*d*t))*0.5*t
        mat[(1,0)] = -1j*Orabi*np.exp(-1j*(np.abs(l)*np.pi*0.5-0.5*d*t))*0.5*t
        mat[(1,1)] = np.exp(1j*0.5*d*t)*(1 - 1j*d*0.5*t)
    else:
        mat[(0,0)] = np.exp(-1j*0.5*d*t)*(np.cos(0.5*f*t) + 1j*d/f*np.sin(0.5*f*t))
        mat[(0,1)] = -1j*Orabi/f*np.exp(1j*(np.abs(l)*np.pi*0.5-0.5*d*t))*np.sin(0.5*f*t)
        mat[(1,0)] = -1j*Orabi/f*np.exp(-1j*(np.abs(l)*np.pi*0.5-0.5*d*t))*np.sin(0.5*f*t)
        mat[(1,1)] = np.exp(1j*0.5*d*t)*(np.cos(0.5*f*t) - 1j*d/f*np.sin(0.5*f*t))
    return(mat)



def mp_helper_essa(t, sb, d, Omega, n_LD, cList, spin_probe, n, dec = 0, lim = 0.5):
    """helper function for parallelizing the calculatino in evolution_spinState_Analytical(...)"""

    # calculate the coefficients c of the state at time t in the fock basis
    cts = [np.matmul(T_l_n(t, sb, i, d, Omega, n_LD), cList[i]) for i in range(n)]

    # get the ground or excited spin state probability
    pgs = [np.abs(ct[1-spin_probe])**2 for ct in cts]
    return(np.exp(-dec*t)*(np.sum(pgs)-lim) + lim)

def evolution_spinState_Analytical(times, spin_init, spin_probe, phonon_init, Omega, n_LD, dec = 0, lim = 0.5, sb = +1, d = 0, parallel = True):
    """calculates the timeevolutin of the occupation probability of one spin state.
    The calculation is based on Formula (83) in LBM+03.
    Returns for each time point in times the occupation probability of the choosen spin state.
    parameters:
        times: timepoints for which the occupation probability should be calculated
        spin_init: initial spin state, 0 for ground state (down), 1 for excited state (up)
        spin_probe: spin state for which the occupation probability should be calculated, 0 ground, 1 excited
        phonon_init: initial phonon state (=> total initial state: |ini> = |spin_init>|phonon_init>)
        sb: which sideband is driven (+1 first blue, -1 first red, 0 carrier, +2 second blue, ...)
        Omega: Rabi frequency
        n_LD: Lamb Dicke parameter
        d: detuning of the laser from sideband (default = 0)
        parallel: if parallel computing should be used (may be much faster)
    """
    # get dimension of Hilbert space
    n = phonon_init.dims[0][0]

    # get the full initial state, cList[i] = (c(i, e), c(i, g)), i phonon fock state, g/e spin ground/excited state)
    if spin_init == 1:
        cList = [[np.sqrt(np.diag(phonon_init.full())[i]), 0j] for i in range(n)]
    elif spin_init == 0:
        cList = [[0j, np.sqrt(np.diag(phonon_init.full())[i])] for i in range(n)]
    else:
        raise ValueError("spin_init must be 1 or 0")

    if spin_probe != 0 and spin_probe !=1:
        raise ValueError("spin_probe must be 1 or 0")

    # get the probabilities for ground/excited state populations
    if parallel:
        # prepare the arguments for the starmap function
        args = [(t, sb, d, Omega, n_LD, cList, spin_probe, n, dec, lim) for t in times]

        # calculate pList using multiprocessing
        if __name__ == "functions_SpinPhonon":
            pool = mp.Pool(mp.cpu_count())
            pList = pool.starmap(mp_helper_essa, args)
            time.sleep(0.1)
            pool.close()
            return(pList)
    else:
        pList = []
        for t in times:
            # calculate the coefficients c of the state at time t in the fock basis
            cts = [np.matmul(T_l_n(t, sb, i, d, Omega, n_LD), cList[i]) for i in range(n)]
            # get the ground or excited spin state probability (remember ct = (c(e), c(g)))
            pgs = [np.abs(ct[1-spin_probe])**2 for ct in cts]
            pList.append(np.exp(-dec*t)*(np.sum(pgs)-lim) + lim)

        return(pList)


def simulate_QPN(values, n_samples, n_skipp = 1):
    """simulates an experiment by doing the projections on one quantum state with each measurement
    parameters:
        values: list of occupation probabilities for the quantum state
        n_samples: number of measurements for each point
        n_skipp: number of values between each point"""

    yVals = []
    yErrs = []
    values = values[::n_skipp]
    for val in values:
        Pys = np.full(n_samples, val)
        rs = np.random.rand(n_samples)
        ys = rs < Pys # check which random numbers are lower than val (keep in mind: True = 1, False = 0)
        yVals.append(np.mean(ys))
        yErrs.append(np.sqrt(np.var(ys)/n_samples))

    return(yVals, yErrs)


def simulate_rb_sb_flop(times, state, Omega, n_LD, n_samples):
    """simulates the red/blue sideband flops for a given state using QPN
    parameters:
        times: list of times, at which datapoints should be simulated
        state: initial phonon state
            (initial spin state is |g> for redflop, |e> for blueflop)
        Omega: Rabi frequency
        n_LD: Lamb Dicke parameter
        n_samples: number of single measurements for each datapoint

    returns:
        redflop: list of times, simulated datapoints and errors (one for each t in times)
        blueflop: the same as redflop
        these are ready to go into the various fit funcitons in eios_sb
    """
    # get the expected ground state probability for red and blue sidebands
    probs_RSB = evolution_spinState_Analytical(times, 0, 0, state, Omega, n_LD, sb = -1, parallel = True)
    probs_BSB = evolution_spinState_Analytical(times, 1, 0, state, Omega, n_LD, sb = 1, parallel = True)

    # simulate QPN
    points_RSB, errs_RSB = simulate_QPN(probs_RSB, n_samples)
    points_BSB, errs_BSB = simulate_QPN(probs_BSB, n_samples)

    # avoiding zeros
    errs_RSB = [np.max([err, 0.0000001]) for err in errs_RSB]
    errs_BSB = [np.max([err, 0.0000001]) for err in errs_BSB]

    # preparing data
    redflop = [times, np.array(points_RSB), np.array(errs_RSB)]
    blueflop = [times, np.array(points_BSB), np.array(errs_BSB)]
    return redflop, blueflop


def diff_fock_state(state, fock, f_ok = True):
    """calculates the L2 difference of the fock states between a state and a fock state distribution
    returns sqrt(sum((n_state - n_fock)^2))
    parameters:
        state: a quantum state given as qutip object
        fock: a fock state distribution of the form [[0, n_0], [1, n_1], ...], should have the same length as state has dimension
        f_ok: bool, checks if the difference should be calculated (return -1 if f_ok = False)
    """
    n = state.dims[0][0]
    if f_ok: # only if fit not failed
        # get fock state distribution for fit
        fock = np.array(fock)[:,1]
        # get fock state distribution for original state
        fock_state = np.abs(np.diag(state.full()))
        # calculate the difference between the two
        return(np.sum([(fock[i] - fock_state[i])**2 for i in range(len(fock))]))
    else: # if fit failed
        return(-1)



def plot_fit_fock_sb_res(times, res, n_LD, compare_state = False, phonon_init = fock_dm(2, 0), compare_data = False, data = [], figsize=(15, 7)):
    """plots the sideband flop for the result of a fit of fock sates, and compares it to the data of the fit, and/or to the sideband flop of some initial state
    parameters:
        times: time points, where the sideband fit should be calculated
        res: result of the fit (should be the full returnvalue of the fit funciton from eios, make sure to fit the fock states, not params)
    optional parameters:
        compare_state: bool, if the sb flop of a state (= phonon_init) should be plotted as well (for comparison)
        compare_data: bool, if the sb flop should be compared to the data which go into the fit
        data: fit data, in the form daa = [redflop, blueflop], redflop = [times, flopdata, floperror]"""

    _,fmin,_,_,_,params,_,_,focks = res

    if not fmin['is_valid']:
        raise ValueError("fit failed")

    Om = params[0]*2*np.pi
    dec = params[1]
    limB = params[2]
    limR = params[3]

    focks = focks[1]
    nFock = len(focks)
    # print(focks)

    psi_re = focks[0] * fock_dm(nFock, 0)
    for i in range(nFock-1):
        psi_re += focks[i+1]*fock_dm(nFock, i+1)

    probs_RSB_re = evolution_spinState_Analytical(times, 0, 0, psi_re, Om, n_LD, dec=dec, lim=limR, sb = -1, parallel = True)
    probs_BSB_re = evolution_spinState_Analytical(times, 1, 0, psi_re, Om, n_LD, dec=dec, lim=limB, sb = 1, parallel = True)

    if compare_state:
        probs_RSB_in = evolution_spinState_Analytical(times, 0, 0, phonon_init, Om, n_LD, dec=dec, lim=limR, sb = -1, parallel = True)
        probs_BSB_in = evolution_spinState_Analytical(times, 1, 0, phonon_init, Om, n_LD, dec=dec, lim=limB, sb = 1, parallel = True)

    plt.figure(figsize = figsize)
    plt.plot(times, probs_RSB_re, "r--", label="RSB, fit")
    plt.plot(times, probs_BSB_re, "b--", label="BSB, fit")

    if compare_state:
        plt.plot(times, probs_RSB_in, "r:", label="RSB, initial state")
        plt.plot(times, probs_BSB_in, "b:", label="BSB, initial state")

    if compare_data:
        plt.errorbar(data[0][0], data[0][1], yerr=data[0][2], fmt = 'ro', capsize = 2, label="RSB, data")
        plt.errorbar(data[1][0], data[1][1], yerr=data[1][2], fmt = 'bo', capsize = 2, label="BSB, data")

    plt.legend()
    plt.show()
