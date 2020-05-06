import numpy as np
from qutip import *
import time

from math import factorial
from scipy.special import eval_genlaguerre

import matplotlib.pyplot as plt

from PyModules.analyse_eios.eios_analyse import fit_direct, fit_poisson_hist, poisson_pdf, fit_hist

amu = 1.66053892099999983243145829272e-27;
hbar = 1.05457172533628960870683374522e-34;
MHz = 1e6;
nm = 1e-9;

def LDparameter(fmode, angle):
    return np.sqrt(2)*np.abs(np.cos(angle/180*np.pi))*2*np.pi/(280*nm)*np.sqrt(hbar/(2*25*amu * 2* np.pi * fmode*MHz))

def OmegaOC(n0, n1, Omega0, LDparameter):
    dn = n1-n0;
    if (dn>0):
        tOmega = Omega0 * np.exp(-0.5*LDparameter**2) * LDparameter**dn * np.sqrt(factorial(n0)/factorial(n1)) * eval_genlaguerre(n0,dn,LDparameter**2)
    elif (dn<0):
        tOmega = Omega0 * np.exp(-0.5*LDparameter**2) * LDparameter**np.abs(dn) * np.sqrt(factorial(n1)/factorial(n0)) * eval_genlaguerre(n1,np.abs(dn),LDparameter**2)
    else:
        tOmega = Omega0 * np.exp(-0.5*LDparameter**2) * eval_genlaguerre(n0,dn,LDparameter**2)
    return tOmega

def OCFlop1Mdec(LDparameter, Fockdist, Rabi0, gamma, lim, dn, tlist):
    values=[];
    for tt in tlist:
        if dn > 0: tsum = 0;
        else: tsum = 1;
        if (dn < 0): n0 = np.abs(dn)
        else: n0 = 0
        for tn in np.linspace(n0,Fockdist[-1][0],len(Fockdist)-n0):
            #trate = np.abs(OmegaOC(int(tn), int(tn+dn), Rabi0, LDparameter));
            trate = OmegaOC(tn, tn+dn, Rabi0, LDparameter);
            if dn > 0:
                tsum = tsum + Fockdist[int(tn)][1]*np.sin(np.pi*trate*tt)**2
            else: tsum = tsum - Fockdist[int(tn)][1]*np.sin(np.pi*trate*tt)**2
        tsumdec = lim + np.exp(-gamma * tt)*(tsum - lim)
        values.append(tsumdec)
    return values

def mixedfockdist(nmax, nthermal, ncoherent, nsqueezed, ntrot):
    alpha = np.sqrt(ncoherent)
    rsqueeze = np.arcsinh(np.sqrt(nsqueezed)) ## update: MW 04.12.2019
    thermalpops = [];
    for n in range(0,nmax+1):
        thermalpops.append([ n , nthermal**n / (1+nthermal)**(n+1) ]);

    ntr = np.abs(ntrot)
    op_sq = squeeze(nmax+1, rsqueeze/ntr) ## update: MW 04.12.2019
    op_dp = displace(nmax+1, alpha/ntr)
    if ntrot > 0:
        op = op_sq*op_dp
    elif ntrot < 0:
        op = op_dp*op_sq
    op_trot = op**ntr

    displacednumberstates = [];
    for n in range(0,nmax+1):
        num_state = op_trot * basis(nmax+1,n);
        displacednumberstates.append((num_state*num_state.dag()).diag());

    outputdist = [];
    for tF in range(0,nmax+1):
        tnpop = [];
        for tn in range(0,nmax+1):
            thermalfact = thermalpops[tn][1];
            tpop = np.real(displacednumberstates[tF][tn]) * thermalfact;
            tnpop.append(tpop);
        outputdist.append([tF, sum(tnpop)]);
    return outputdist

def normalizefockdist(fockdist):
    tsum = sum([item[1] for item in fockdist])
    normalizedfockdist = []
    for item in fockdist:
        itemnorm = [item[0],item[1]/tsum]
        normalizedfockdist.append(itemnorm)
    return normalizedfockdist

from iminuit import Minuit

def fit_flop_sb_fock(redflop, blueflop, LD, nmax, initparams, fixparams):
    [tdatabsb, flopdatabsb, errsbsb] = blueflop;
    [tdatarsb, flopdatarsb, errsrsb] = redflop;

    def fit_function_freepops(par):
        Rabi = par[0]
        dec = par[1]
        limb = par[2]
        limr = par[3]
        fockdist = []
        for tp in range(4,len(par)):
            fockdist.append([tp-4,par[tp]])

        modelBSB = OCFlop1Mdec(LD, fockdist, Rabi, dec, limb, 1, tdatabsb);
        modelRSB = OCFlop1Mdec(LD, fockdist, Rabi, dec, limr, -1, tdatarsb);

        residBSB = (flopdatabsb - modelBSB)**2/errsbsb**2
        residRSB = (flopdatarsb - modelRSB)**2/errsrsb**2
        return sum(residBSB)+sum(residRSB)

    Rabi_init = initparams[0]
    dec_init = initparams[1]
    limb_init = initparams[2]
    limr_init = initparams[3]
    p0_init = 0.9
    p1_init = 0.05
    p2_init = 0.05
    Rabi_error = 0.0
    dec_error = 0.0
    limb_error = 0.0
    limr_error = 0.0
    pn_error = 0.0

    initvals = [Rabi_init,dec_init,limb_init,limr_init]
    errorvals = [Rabi_error,dec_error,limb_error,limr_error]
    limits = [(0,None),(0,None),(0,1),(0,1)]
    names = ["Rabi","dec","limb","limr"]
    fixes = [fixparams[0],fixparams[1],fixparams[2],fixparams[3]]
    for i in range(4,nmax+4):
        initvals.append(0.1)
        errorvals.append(pn_error)
        names.append('p%i'%(i-4))
        limits.append(tuple([0.,1.]))
        fixes.append(0)

    n_param = np.sum(np.array(fixes)<1.)
    print('Free parameter count',n_param)

    m = Minuit.from_array_func(fit_function_freepops,
                           tuple(initvals), error = tuple(errorvals), fix = tuple(fixes),
                           limit = tuple(limits), name = tuple(names),
                           errordef=1)

    # fit it
    print('migrad started at',time.asctime( time.localtime(time.time()) ) )
    fmin, param = m.migrad(ncall=1000000);
    print('migrad finished at',time.asctime( time.localtime(time.time()) ) )
    red_chi = fmin.fval / (len(flopdatabsb)+len(flopdatarsb) - n_param)
    #print(red_chi)
    fit_rabi=m.values[0]
    fit_dec=m.values[1]
    fit_limb=m.values[2]
    fit_limr=m.values[3]

    fit_rabi_err=m.errors[0]
    fit_dec_err=m.errors[1]
    fit_limb_err=m.errors[2]
    fit_limr_err=m.errors[3]

    fit_fockdist = []
    for tn in range(4,len(m.values)):
        fit_fockdist.append([tn-4,m.values[tn]])

    fit_fockdist_norm = normalizefockdist(fit_fockdist)
    fit_fock_n = [item[0] for item in fit_fockdist_norm]
    fit_fock_p = [item[1] for item in fit_fockdist_norm]

    fit_fockdist_sum = sum([item[1] for item in fit_fockdist])
    #print(m.values)
    #print(m.errors)
    fit_fock_e = []
    for tn in range(4,len(m.errors)):
        fit_fock_e.append(m.errors[tn]/fit_fockdist_sum)

    flop_func_rsb = lambda t: OCFlop1Mdec(LD, fit_fockdist_norm, fit_rabi, fit_dec, fit_limr, -1, t)
    flop_func_bsb = lambda t: OCFlop1Mdec(LD, fit_fockdist_norm, fit_rabi, fit_dec, fit_limb, 1, t)
    flop_func_list = [flop_func_rsb, flop_func_bsb]

    return red_chi, fmin, param, m,\
            flop_func_list, \
            [fit_rabi, fit_dec, fit_limb, fit_limr], \
            [fit_rabi_err, fit_dec_err, fit_limb_err, fit_limr_err], \
            fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e]

def fit_dist_fock(fock_n, fock_p, fock_e, initparams, fixparams, ntrot):
    nmax = len(fock_n)-1

    def fit_function_fockdist(par):
        [nth, ncoh, nsq] = par
        modeldist = mixedfockdist(nmax, nth, ncoh, nsq, ntrot)
        tsum = 0.
        for tn in fock_n:
            tsum += (modeldist[tn][1] - fock_p[tn])**2/(fock_e[tn]**2)
        return tsum

    errorvals = [0.001,0.001,0.001]
    limits = [(0,None),(0,None),(0,None)]
    names = ["nth","ncoh","nsq"]

    n_param = np.sum(np.array(fixparams)<1.)
    print('Free parameter count',n_param)

    m = Minuit.from_array_func(fit_function_fockdist,
                           tuple(initparams), error = tuple(errorvals), fix = tuple(fixparams),
                           limit = tuple(limits), name = tuple(names),
                           errordef=1)

    fmin, param = m.migrad(ncall=100000);
    red_chi = fmin.fval / (len(fock_n) - n_param)

    [fit_nth, fit_ncoh, fit_nsq] = m.values.values()
    [fit_nth_err, fit_ncoh_err, fit_nsq_err] = m.errors.values()

    fit_fockdist_norm = mixedfockdist(nmax, fit_nth, fit_ncoh, fit_nsq, ntrot)
    fit_fock_n = [item[0] for item in fit_fockdist_norm]
    fit_fock_p = [item[1] for item in fit_fockdist_norm]

    return red_chi, fmin, param, m,\
            m.values.values(), m.errors.values(), \
            fit_fockdist_norm, [fit_fock_n, fit_fock_p]

def fit_flop_sb(redflop, blueflop, LD, nmax, initparams, fixparams, ntrot):
    [tdatabsb, flopdatabsb, errsbsb] = blueflop;
    [tdatarsb, flopdatarsb, errsrsb] = redflop;

    def fit_function_par(par):
        [Rabi, dec, limb, limr, nth, ncoh, nsq] = par
        fockdistnorm = mixedfockdist(nmax, nth, ncoh, nsq, ntrot)

        modelBSB = OCFlop1Mdec(LD, fockdistnorm, Rabi, dec, limb, 1, tdatabsb);
        modelRSB = OCFlop1Mdec(LD, fockdistnorm, Rabi, dec, limr, -1, tdatarsb);

        residBSB = (flopdatabsb - modelBSB)**2/errsbsb**2
        residRSB = (flopdatarsb - modelRSB)**2/errsrsb**2
        return sum(residBSB)+sum(residRSB)
    #print(initparams)
    Rabi_error = 0.01
    dec_error = 0.0001
    limb_error = 0.01
    limr_error = 0.01
    nth_error = 0.001
    ncoh_error = 0.001
    nsq_error = 0.001
    errorvals=[Rabi_error,dec_error,limb_error,limr_error,nth_error,ncoh_error,nsq_error]
    rabi=initparams[0]
    limits = [(0.5*rabi,1.5*rabi),(0.05*rabi,.2*rabi),(0.,0.7),(0.0,nmax),(0,nmax),(0,nmax),(0,None)]
    names = ["Rabi","dec","limb","limr","nth","ncoh","nsq"]

    n_param = np.sum(np.array(fixparams)<1.)
    print('Free parameter count',n_param)

    m = Minuit.from_array_func(fit_function_par,
                           tuple(initparams), error=tuple(errorvals), fix=tuple(fixparams),
                           limit=tuple(limits), name=tuple(names),
                           errordef=1)

    # fit it
    print('migrad started at',time.asctime( time.localtime(time.time()) ) )
    fmin, param = m.migrad(ncall=100000);
    print('migrad finished at',time.asctime( time.localtime(time.time()) ) )
    red_chi = fmin.fval / (len(flopdatabsb)+len(flopdatarsb) - n_param)

    [fit_rabi, fit_dec, fit_limb, fit_limr, fit_nth, fit_ncoh, fit_nsq] = m.values.values()
    [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err,fit_nth_err,fit_ncoh_err,fit_nsq_err] = m.errors.values()

    fit_fockdist_norm = mixedfockdist(nmax, fit_nth, fit_ncoh, fit_nsq, ntrot)
    fit_fock_n = [item[0] for item in fit_fockdist_norm]
    fit_fock_p = [item[1] for item in fit_fockdist_norm]
    fit_fock_e = [0]*len(fit_fock_n)
    flop_func_rsb = lambda t: OCFlop1Mdec(LD, fit_fockdist_norm, fit_rabi, fit_dec, fit_limr, -1, t)
    flop_func_bsb = lambda t: OCFlop1Mdec(LD, fit_fockdist_norm, fit_rabi, fit_dec, fit_limb, 1, t)
    flop_func_list = [flop_func_rsb, flop_func_bsb]

    return red_chi, fmin, param, m, \
            flop_func_list, m.values.values(), m.errors.values(), \
            fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e]

def fit_flop_carrier(flop, LD, nmax, initparams, fixparams, ntrot):
    [tdata, flopdata, errs] = flop;

    def fit_function_par(par):
        [Rabi, dec, lim, nth, ncoh, nsq] = par
        fockdistnorm = mixedfockdist(nmax, nth, ncoh, nsq, ntrot)
        model = OCFlop1Mdec(LD, fockdistnorm, Rabi, dec, lim, 0, tdata);
        resid = (flopdata - model)**2/errs**2
        return sum(resid)

    Rabi_error = 0.01
    dec_error = 0.0001
    lim_error = 0.01
    nth_error = 0.001
    ncoh_error = 0.001
    nsq_error = 0.001
    errorvals=[Rabi_error,dec_error,lim_error,nth_error,ncoh_error,nsq_error]

    limits = [(0,1),(0,None),(0,1),(0,None),(0,None),(0,None)]
    names = ["Rabi","dec","limb","nth","ncoh","nsq"]

    n_param = np.sum(np.array(fixparams)<1.)
    print('Free parameter count',n_param)

    m = Minuit.from_array_func(fit_function_par,
                           tuple(initparams), error = tuple(errorvals), fix = tuple(fixparams),
                           limit = tuple(limits), name = tuple(names),
                           errordef=1)

    # fit it
    print('migrad started at',time.asctime( time.localtime(time.time()) ) )
    fmin, param = m.migrad(ncall=100000);
    print('migrad finished at',time.asctime( time.localtime(time.time()) ) )
    red_chi = fmin.fval / (len(flopdata) - n_param)

    [fit_rabi, fit_dec, fit_lim,  fit_nth, fit_ncoh, fit_nsq] = m.values.values()
    [fit_rabi_err,fit_dec_err,fit_lim_err,fit_nth_err,fit_ncoh_err,fit_nsq_err] = m.errors.values()

    fit_fockdist_norm = mixedfockdist(nmax, fit_nth, fit_ncoh, fit_nsq, ntrot)
    fit_fock_n = [item[0] for item in fit_fockdist_norm]
    fit_fock_p = [item[1] for item in fit_fockdist_norm]
    fit_fock_e = [0]*len(fit_fock_n)
    flop_func = lambda t: OCFlop1Mdec(LD, fit_fockdist_norm, fit_rabi, fit_dec, fit_lim, 0, t)
    flop_func_list = [flop_func]

    return red_chi, fmin, param, m, \
            flop_func_list, m.values.values(), m.errors.values(), \
            fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e]


def fit_RSB_BSB(redflop, blueflop, LD, nmax, initparams, fixparams, doplot=False, lbl_list=None, figsize=(9,4.5)):
    red_chi, fmin, param, m, flop_func_list, \
    [fit_rabi, fit_dec, fit_limb, fit_limr], \
    [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err], \
    fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = fit_flop_sb_fock(redflop, blueflop, LD, nmax, initparams, fixparams)

    if doplot:
        fit_valid = fmin['is_valid']
        if fit_valid:
            fit_status = 'fock state distribution'
        else:
            fit_status = 'fit failed'
        if lbl_list is None:
            lbl_list = ['red sideband flop', 'blue sideband flop']
        plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [redflop, blueflop], lbl_list, fit_status, figsize);
        plt.show()

    return [fit_fock_n, fit_fock_p, fit_fock_e, fit_rabi, fit_dec, fit_limr, fit_limb, fit_fockdist_norm]

# initparams=[Rabi_init,dec_init,limb_init,limr_init,nth_init,ncoh_init,nsq_init]
def fit_RSB_BSB_par(redflop, blueflop, LD, nmax, initparams, fixparams, ntrot, doplot=False, lbl_list=None, figsize=(9,4.5)):
    #print(LD, nmax, initparams, fixparams, ntrot, doplot, lbl_list, figsize)
    red_chi, fmin, param, m, flop_func_list, \
        [fit_rabi, fit_dec, fit_limb, fit_limr, fit_nth, fit_ncoh, fit_nsq], \
        [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err,fit_nth_err,fit_ncoh_err,fit_nsq_err], \
        fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = \
            fit_flop_sb(redflop, blueflop, LD, nmax, initparams, fixparams, ntrot)

    if doplot:
        fit_valid = fmin['is_valid']
        if fit_valid:
            fit_status = '$red. \chi^2$= %.3f\n $\Omega_{0}$= %.3f +- %.3f\n $\Gamma_{dec}$= %.3f +- %.3f\n $n_{th}$= %.3f +- %.3f\n$n_{coh}$= %.3f +- %.3f\n$n_{sq}$= %.3f +- %.3f' % (red_chi, fit_rabi, fit_rabi_err, fit_dec, fit_dec_err, fit_nth, fit_nth_err, fit_ncoh, fit_ncoh_err, fit_nsq, fit_nsq_err)
        else:
            fit_status = 'fit failed'
        if lbl_list is None:
            lbl_list = ['RSB', 'BSB']
        plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [redflop, blueflop], lbl_list, fit_status, figsize);
        plt.subplots_adjust(wspace=0.12)
        plt.show()

        #print(fit_status)

    return red_chi, fmin, param

def fit_carrier_par(flop, LD, nmax, initparams, fixparams, ntrot, doplot=False, lbl_list=None, figsize=(9,4.5)):
    red_chi, fmin, param, m, flop_func_list, \
        [fit_rabi, fit_dec, fit_lim, fit_nth, fit_ncoh, fit_nsq], \
        [fit_rabi_err,fit_dec_err,fit_lim_err,fit_nth_err,fit_ncoh_err,fit_nsq_err], \
        fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = \
            fit_flop_carrier(flop, LD, nmax, initparams, fixparams, ntrot)

    if doplot:
        fit_valid = fmin['is_valid']
        if fit_valid:
            fit_status = '$n_{th}$\t= %.4f +- %.4f\n$n_{coh}$\t= %.4f +- %.4f\n$n_{sq}$\t= %.4f +- %.4f' % (fit_nth,fit_nth_err,fit_ncoh,fit_ncoh_err,fit_nsq,fit_nsq_err)
        else:
            fit_status = 'fit failed'
        if lbl_list is None:
            lbl_list = ['red sideband flop', 'blue sideband flop']

        plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [flop], lbl_list, fit_status, figsize);
        plt.show()

        print(fit_status)

    return red_chi, fmin, param

def fit_fockdist(ns, pns, pnerrs, initparams, fixparams, ntrot, do_plot):
    red_chi, fmin, param, m, \
        [fit_nth, fit_ncoh, fit_nsq], \
        [fit_nth_err, fit_ncoh_err, fit_nsq_err], \
        fit_fockdist_norm, [fit_fock_n, fit_fock_p] = \
            fit_dist_fock(ns, pns, pnerrs, initparams, fixparams, ntrot)

    output = np.array([[fit_nth,fit_nth_err],[fit_ncoh,fit_ncoh_err],[fit_nsq,fit_nsq_err],fit_fockdist_norm])

    if do_plot == True:
        plot_fock_fit(ns, pns, pnerrs, fit_fock_n, fit_fock_p)
        plt.show()
    return output

def plot_fock_fit(fock_n, fock_p, fock_e, fit_fock_n, fit_fock_p, figsize=(9,4.5)):
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize);
    ax.bar(fit_fock_n, fit_fock_p, width=0.8, align = 'center');
    ax.errorbar(fock_n, fock_p, yerr=fock_e, marker='o', linestyle='None', color=cmap(1));
    ax.set_xlabel('Fock state')
    ax.set_ylabel('Population')
    ax.set_ylim(0,1);

def plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, flop_list, lbl_list, fit_status, figsize=(9,4.5)):
    fig, ax = plt.subplots(2, 2, figsize=figsize, gridspec_kw = {'width_ratios':[3, 1],'height_ratios':[4, 1]});
    ColorList=['red','navy','orange','grey','silver','black']
    t_min = []
    t_max = []
    #t_flop_fine = np.linspace(np.min(t_min),np.max(t_max),1000)
    i=0
    for flop, lbl, func in zip(flop_list, lbl_list, flop_func_list):
        [t_flop, y_flop, err_flop] = flop
        t_min.append(np.min(t_flop))
        t_max.append(np.max(t_flop))
        ax[0,0].errorbar(t_flop, y_flop, yerr=err_flop, linestyle="None", marker="o", markersize=7.5, lw=1., capsize=.0, label=lbl, color=ColorList[i])
        ax[0,0].plot(t_flop,func(t_flop), color=ColorList[i], linestyle='-', linewidth=3.)
        #ax[0,0].fill_between(t_flop, 1-i, func(t_flop), color=ColorList[i], interpolate=True, alpha=.3)
        ax[1,0].errorbar(t_flop, np.array(y_flop)-np.array(func(t_flop)), yerr=err_flop, linestyle="None", marker="o", markersize=7.5, lw=1., capsize=.0, label=lbl, color=ColorList[i])
        i=i+1

    #t_flop_fine = np.linspace(np.min(t_min),np.max(t_max),1000)
    #i=0
    #for func in flop_func_list:
    #    ax[0,0].plot(t_flop_fine,func(t_flop_fine), color=ColorList[i], linestyle='-', linewidth=2)
    #    ax[0,0].fill_between(t_flop_fine, 1-i, func(t_flop_fine), color=ColorList[i], interpolate=True, alpha=.3)
    #    ax[1,0].errorbar(t_flop, np.array(y_flop)-np.array(func(t_flop)), yerr=err_flop, linestyle="None", marker="o", markersize=7.5, lw=1., capsize=.0, label=lbl, color=ColorList[i])
    #    i=i+1


    ax[1,0].set_xlabel('Coupling duration ($\mu s$)', fontsize=12)
    ax[0,0].set_ylabel('$P_\downarrow$', fontsize=12)
    ax[0,0].set_ylim(-0.025,1.025);
    ax[1,0].set_ylim(-.2,.2);
    ax[1,0].set_ylabel('Res.', fontsize=12)
    #ax[0,0].legend()

    #cmap = plt.get_cmap("tab10")
    #ax[1].errorbar(fit_fock_n, fit_fock_p, yerr=fit_fock_e, marker='None', linestyle='None', color=cmap(1), label=fit_status)
    ax[0,1].errorbar(fit_fock_n, fit_fock_p, yerr=fit_fock_e, linestyle="None", marker="o", markersize=7.5, lw=1., capsize=.0, label=lbl, color='black')
    ax[0,1].bar(fit_fock_n, fit_fock_p, yerr=fit_fock_e, width=0.75, align='center', alpha=0.3, capsize=.0, label=fit_status, color='orange')
    ax[1,1].set_xlabel('Fock state', fontsize=12)
    ax[0,1].set_ylabel('Population', fontsize=12)
    #ax[1,1].set_ylabel('Res.', fontsize=12)
    ax[0,1].set_ylim(0,1);
    ax[1,1].set_ylim(-.2,.2);
    ax[0,1].legend(loc=2, bbox_to_anchor=(.5, 1.))
    return fig, ax

import pickle
import os.path
from PyModules.analyse_eios.eios_data import read, read_xml
from PyModules.analyse_eios.eios_analyse import fit_direct

def open_file(filename, cache_path):
    q_fn = []
    q_sn = []
    for first_fn in filename:
        pos_1 = first_fn.rfind('/')
        pos_2 = first_fn[:pos_1].rfind('/')+1
        q_fn.append(first_fn[pos_1+1:first_fn.rfind('.dat')])
        q_sn.append(first_fn[pos_2:pos_1])
        print('%s: %s' %(q_sn[-1],q_fn[-1]))

    quick_name = cache_path+'%s.pkl'%q_fn[0]
    if os.path.isfile(quick_name):
        flops = pickle.load( open( quick_name, 'rb') )
        redflop = flops[0]
        blueflop = flops[1]
    else:
        # two files for red & blue sideband
        if len(filename)==2:
            for name in filename:
                data, xml = read(name, sort=True, skip_first=False)
                xml_dict = read_xml(xml)
                ionprop = xml_dict['ionproperties']
                init_spin = ionprop['A_init_spin_up']
                x = data[0]['x']
                hists = data[0]['hists']

                print('spin init = %i; %s sb'%(init_spin,'red'*int(1-init_spin) + 'blue'*int(init_spin)))

                y, y_err = fit_direct(hists)

                if init_spin>0: #blue
                    blueflop = [x,y,y_err]
                else: #red
                    redflop = [x,y,y_err]
        # single file with two counter
        elif len(filename)==1:
            data, xml = read(filename[0], sort=True, skip_first=False)
            xml_dict = read_xml(xml)

            hists_b=data[0]['hists']
            hists_r=data[1]['hists']
            max_avg_photons=int(np.max(np.asarray([data[1]['y'],data[0]['y']]).flatten()))
            min_avg_photons=int(np.min(np.asarray([data[1]['y'],data[0]['y']]).flatten()))

            #All in one histogram
            hist_b=np.asarray(hists_b).flatten()
            hist_r=np.asarray(hists_r).flatten()
            hist_comb=np.asarray([hist_b, hist_r]).flatten()

            pre_fit = fit_poisson_hist(hist_comb, min_avg_photons, max_avg_photons)
            mu1,mu2,pup=pre_fit['x']

            if len(data)>1:
                x = data[0]['x']
                y, y_err = fit_direct(hists_b, pre_fit=pre_fit)
                blueflop = [x,y,y_err]

                x = data[1]['x']
                y, y_err = fit_direct(hists_r, pre_fit=pre_fit)
                redflop = [x,y,y_err]
            else:
                raise Exception('Expecting data of two counter!')
        else:
            raise Exception('Expecting two files (w/ one counter data) or one file (w/ two counter data)!')

        print('Save data locally %s'%quick_name)
        pickle.dump([redflop,blueflop], open(quick_name, 'wb') )

    if len(filename)<2:
        q_sn.extend(q_sn)
        q_fn.extend(q_fn)
    lbl = [x+': '+y for x,y in zip(q_sn,q_fn)]
    return redflop, blueflop, lbl
