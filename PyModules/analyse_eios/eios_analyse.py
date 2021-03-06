import numpy as np
import math
import iminuit
import inspect

# included from rob
import multiprocessing as mp
import time


import matplotlib.pyplot as plt

from scipy.stats import poisson
# from scipy.stats.chi2 import cdf as chi2cdf
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from probfit import UnbinnedLH

def significant_digit(x):
    if math.isnan(x) or math.isinf(x):
        return 0
    try:
        y = -int(math.floor(math.log10(abs(x))))
    except:
        y = 0
    return y

def round_sig(par,par_err):
    for i in range(min([len(par),len(par_err)])):
        sd = significant_digit(par_err[i])
        par[i] = round(par[i],sd)
        par_err[i] = round(par_err[i],sd)
    return par, par_err
############ two ions #################
def mLL_2I(data, mu_dd, mu_du, mu_uu, p_dd, p_du, p_uu):
    """returns the negative log likelihood functin of a three-poissonian distribution with:
        means mu_dd, mu_du, mu_uu
        weights  p_dd, p_du, p_uu
        data: an array filled with random numbers from the three-poissonian distribution
    """

    list = np.log(p_dd/(p_dd+p_du+p_uu)*poisson.pmf(data, mu_dd)
                        + p_du/(p_dd+p_du+p_uu)*poisson.pmf(data, mu_du)
                        + p_uu/(p_dd+p_du+p_uu)*poisson.pmf(data, mu_uu))

    # remove inf's
    if True:
        l = np.isinf(list)
        list = np.array(list)[~np.isinf(list)]

    return(-np.sum(list) + np.sum(l)*np.abs(np.sum(list)))


def fit_poisson_from_file_2I(path, prefit = None, ret_prefit = False, onlyFirst = False):
    """fits three poissonians on the histograms
    parameters:
        path: directory of file for which the poisson fits should be calculated
        prefit: you can give a prefit, to set the count levels for both bright, both dark, one bright one dark
        ret_prefit: if the prefit should be returned (instead of the population probabilities)
        onlyFirst: considers only the first measurement series (is for example in the MW Flop the case)

    returns:
        for each measurement series:
            x, y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu

            x: x-values corresponding to following data_ctr
            y_dd, y_du, y_uu: probability for both ions bright, one dark one bright, both ions dark
            y_err_dd, y_err_du, y_err_uu: corresponding errors (they are way too large!!!)
    """
    data = eios_data.read(path)

    n_data = len(data[0])

    if onlyFirst:
        hists = data[0][0]['hists']
    else:
        hists = [data[0][i]['hists'] for i in range(n_data)]

    if prefit == None:
        MasterHist = np.append([], hists)
        prefit = fit_poisson_hist_2I(MasterHist)
        if ret_prefit:
            return prefit

    res = []
    if onlyFirst:
        x = data[0][0]['x']
        y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu = fit_hist_2I(hists, prefit)
        res.append([x, y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu])
    else:
        for i in range(n_data):
            x = data[0][i]['x']
            y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu = fit_hist_2I(hists[i], prefit)

            res.append([x, y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu])

    return res


def fit_poisson_hist_2I(hist, lowcount=1., highcount=16.):
    """fits a sum of two two-poissonian distributions (see mLL) to a sample hist
    """
    # if hist is a list of histograms, merge them into one hist
    if len(np.shape(hist)) > 1:
        hist = np.append([],hist)

    def func(mu_dd, mu_du, mu_uu, p_dd, p_du, p_uu):
        return mLL_2I(np.array(hist), mu_dd, mu_du, mu_uu, p_dd, p_du, p_uu)

    m = iminuit.Minuit(func, mu_dd = highcount, mu_du = 0.5*highcount, mu_uu = lowcount, p_dd = 0.3, p_du = 0.4, p_uu = 0.3,
                # initial stepsize
               error_mu_dd = 1., error_mu_du = 0.5, error_mu_uu = 0.1, error_p_dd = 0.05, error_p_du = 0.05, error_p_uu = 0.05,
               errordef = 0.5,
                # bounds
               limit_mu_dd = (0., 50), limit_mu_du = (0., 50), limit_mu_uu = (0., 50.), limit_p_dd=(0.,1.), limit_p_du=(0.,1.), limit_p_uu=(0.,1.))
    m.migrad()

    # make sure the probabilities add up to 1
    norm = m.values[3] + m.values[4] + m.values[5]

    res = {
        'x': [m.values[0], m.values[1], m.values[2], m.values[3]/norm, m.values[4]/norm, m.values[5]/norm],
        'x_err': [m.errors[0], m.errors[1], m.errors[2], m.errors[3]/norm, m.errors[4]/norm, m.errors[5]/norm]
    }
    return res

def helper_fit_hist_2I(hist, fit_mu_dd, fit_mu_du, fit_mu_uu, as_err = False):
    """fits the weights of a three-poissonian distribution (with fixed means fit_mu_dd, fit_mu_du, fit_mu_uu) to the sample given in hists
    returns the weight and its error
    """
    if len(hist) == 0: # make sure the hist is not empty
        raise ValueError('empty histogram')
    else:
        # make a helper function to throw into minuit
        # important: since one population is determined by the other two, you need a parameter transformation so that you only have two parameters to get realistic errors
        # to make the matter even more difficult, you need to choose the two parameters in a way, that you make sure that the sum of populations is always 1
        # AND each single probability is between 0 and 1
        # I chose: p_same: probability of both ions being in the same state (= p_uu + p_dd), p_up: probability of being in the up up state, if they are in the same state (p_uu = p_same * p_up)
        func = lambda p_same, p_up: mLL_2I(np.array(hist), fit_mu_dd, fit_mu_du, fit_mu_uu, p_same*(1-p_up), (1-p_same), p_same*p_up)

        # make a minuit object
        m = iminuit.Minuit(func, p_same = 0.6, p_up = 0.3,
                           error_p_same = 0.05, error_p_up = 0.05, errordef = 0.5,
                           limit_p_same=(0.,1.), limit_p_up=(0.,1.))

        # minimize the funciton
        m.migrad()

        # calculate the populations p_dd, p_du, p_uu
        p_dd, p_du, p_uu = m.values[0]*(1-m.values[1]), (1-m.values[0]), m.values[0]*m.values[1]

        if as_err:

            m.minos()
            err = m.np_merrors()

            # print(m.minos())
            # print(err)
            err_s, err_u = [err[0][0], err[1][0]], [err[0][1], err[1][1]]
            P_s, P_u = m.values[0], m.values[1]

            err_uu = [P_u*err_s[0] + P_s*err_u[0] - err_s[0]*err_u[0], P_u*err_s[1] + P_s*err_u[1] - err_s[1]*err_u[1]]
            err_dd = [err_s[0] - P_u*err_s[0] + P_s*err_u[1] - err_s[0]*err_u[1], err_s[1] - P_u*err_s[1] + P_s*err_u[0] + err_s[1]*err_u[0]]
            err_du = [err_s[1], err_s[0]]

        # make sure the probabilities add up to 1
        norm = p_dd + p_du + p_uu
        if np.abs(1-norm) > 0.01:
            print("probabilities not normalized!!!")

        # calculate the errors of p_dd, p_du, p_uu
        if not as_err:
            # print('not')
            err_dd = np.sqrt((m.errors[0]*(1-m.values[1]))**2 + (m.values[0]*m.errors[1])**2)
            err_du = m.errors[0]
            err_uu = np.sqrt((m.errors[0]*m.values[1])**2 + (m.values[0]*m.errors[1])**2)

        # return the parameter and error
        return p_dd, p_du, p_uu, err_dd, err_du, err_uu


# my function for all hist fits (needs pre fit), from rob
def fit_hist_2I(hists, pre_fit, parallel = True, as_err = False):
    t1 = time.time()
    """fits the weights of a three-poissonian distribution to the samples given in hists
    parameters:
        hists: array of arrays, each one a sample of a three-poissonian distribution
        pre_fit: fit result from a fit on all histograms in hists combined
            (the expectation values mu_dd, mu_du, mu_uu are taken and fixed in the fits done here)
            should be a dict. with 'x': [mu_dd, mu_du, mu_uu]
        parallel: bool, default True, if parallel computing should be used (better performance, if the mp module works)
        remove_nan: bool, default True, sets if nans (from empty lists in hists) should be removed in the output
    returns y, y_err
        y: a list of weights for the three-poissonian distribution (or probabilities of being in the corresponding state)
            list of [p_dd, p_du, p_uu]
        y_err: errors of the weights/probabilities
    """

    # take variables from prefit (make sure the expectation values of the poissonians are properly sorted)
    fit_mu_1, fit_mu_2, fit_mu_3,_,_,_ = pre_fit['x']
    [fit_mu_uu, fit_mu_du, fit_mu_dd] = np.sort([fit_mu_1, fit_mu_2, fit_mu_3])

    if parallel:
        args = [(hist, fit_mu_dd, fit_mu_du, fit_mu_uu, as_err) for hist in hists]
        pool = mp.Pool(mp.cpu_count())
        res_y = pool.starmap(helper_fit_hist_2I, args)
        time.sleep(0.01)
        pool.close()
    else:
        res_y = []
        for hist in hists:
            res_y.append(helper_fit_hist_2I(hist, fit_mu_dd, fit_mu_du, fit_mu_uu, as_err))

    # unpack results
    y_dd, y_du, y_uu = [i[0] for i in res_y], [i[1] for i in res_y], [i[2] for i in res_y]
    y_err_dd, y_err_du, y_err_uu = [i[3] for i in res_y], [i[4] for i in res_y], [i[5] for i in res_y]

    # print("fit hists:", np.round(time.time()-t1, 3))
    return y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu


############ FUNCTIONS ############

def func_decay_exponential(x, tau, exponent):
    return np.exp(-(x/tau)**exponent)

def func_decay_reciprocal(x, bndwdth):
    return (1./x)+bndwdth*1e-6

def poisson_pdf(x, p, mu1, mu2):
    return (1.-p)*poisson.pmf(x, mu1)+p*poisson.pmf(x, mu2)

# from rob
def mLL(data, mu1, mu2, p_up):
    """returns the negative log likelihood functin of a two-poissonian distribution with:
        means mu1 and mu2
        weights (1-p_up) and fit_p_up
        data: an array filled with random numbers from the two-poissonian distribution
    """
    # make sure that p_up \in [0, 1]
    if not (0<= p_up <= 1):
        p_up = np.max([p_up, 0])
        p_up = np.min([p_up, 1])

    return -np.sum(np.log((1-p_up)*poisson.pmf(data, mu1) + p_up*poisson.pmf(data, mu2)))

def sin_squared(x, A, freq, phi, gamma):
    x = np.array(x)
    dec = np.exp(-gamma*x)
    osc = np.sin(2*np.pi*x*freq+phi)**2
    return (dec*(.5-A*osc)+.5)

def flop(x, A, freq, phi, dec, C):
    x = np.array(x)
    dec = np.exp(-dec*x)
    osc = np.cos(2*np.pi*x*freq+phi)
    return A*dec*osc+C

def flop_delayed(t, A, freq, phi, dec, C, t0):
    t = np.array(t)
    td = (t>t0)
    y0 = np.logical_not(td)*(A+C)

    t += t0
    dec = np.exp(-dec*t)
    osc = np.cos(2*np.pi*t*freq+phi)
    y1 = td*(A*dec*osc+C)
    return y0+y1

def phase(x, A, multi, phi, C):
    x = np.array(x)
    osc = np.cos(2*x*multi+phi)
    return A*osc+C

def gauss(x, C, A, sigma, x0):
    x = np.array(x)
    u = (x-x0)/sigma
    return A*np.exp(-np.log(2)*u**2)+C

def lorentz(x, C, A, gamma, x0):
    x = np.array(x)
    u = (x-x0)**2
    return A/(np.pi*gamma)*(gamma**2/(u+gamma**2))+C

def sincSquare(x, C, A, sigma, mu):
    x = np.array(x)
    xx = (x[:]-mu)/sigma
    return A*(np.sinc(xx)**2)+C

def sincAbs(x, C, A, sigma, mu):
    x = np.array(x)
    xx = (x[:]-mu)/sigma
    return A*(abs(np.sinc(xx)))+C

def sincSquare_mod(x, C, A, sigma, mu):
    return sincSquare(x, C, -A, sigma, mu)

def parabola(x, C, A, sigma, x0):
    return A*(1-((x-x0)/sigma)**2)+C

def parabola_pos(x,C, A, sigma, x0):
    y = parabola(x,A,x0,sigma,0.)
    for i in range(len(y)):
        y[i] = max(y[i],0.)
    return y+C

def func_lin(x,m,b):
    return m*(x)+b

def func_abs(x,A,x0,C):
    return A*abs(x-x0)+C

def func_sq(x,sigma,x0):
    return parabola(x,1.,x0,sigma,0.)

def func_erf(x,A,x0,sigma):
    return A*erf((x-x0)/sigma)

### SUM ###

def func_sum(func, x, C, *argv):
    k = len(argv)//3
    A = argv[0:k]
    SIGMA = argv[k:2*k]
    X0 = argv[2*k:3*k]
    y = 0.
    for (a,x0,sigma) in zip(A, X0, SIGMA):
        y += func(x,a,x0,sigma,C)
    return y

#def gauss_sum(x, C, *[A], *[SIG], *[X0]):
def gauss_sum(x, C, *argv):
    return func_sum(gauss, x, C, *argv)

def lorentz_sum(x, C,  *argv):
    return func_sum(lorentz,x, C,  *argv)

def sinc_sum(x, C,  *argv):
    return func_sum(sincSquare,x, C,  *argv)

def parabola_sum(x, C,  *argv):
    return func_sum(parabola, x, C,  *argv)

def abs_sum(x, C, *argv):
    return func_sum(func_abs, x, C, *argv)

def sinc_abs_sum(x, C, *argv):
    return func_sum(sincAbs, x, C, *argv)

#######################################

def find_peaks(data,width = 3, strict=True):
    peaks = list()
    for i,v in enumerate(data):
        idx_low = np.arange(i-width,i,1)
        idx_high = np.arange(i+1,i+width+1,1)
        idx = np.concatenate((idx_low,idx_high))
        idx = idx[idx>-1]
        idx = idx[idx<len(data)]

        window = data[idx]-v
        if strict:
            chk_peak = all(window<0)
        else:
            chk_peak = all(window<=0)
        if chk_peak:
            peaks.append(i)

    return peaks

def unpack_sorted(data):
    [x,y,y_err] = data
    idx = np.argsort(x)
    x = x[idx]; y = y[idx]; y_err = y_err[idx]
    return x,y,y_err





# # my function for all hist fits (needs pre fit), from rob
# def fit_hist_2I(hists, pre_fit, parallel = True):
#     t1 = time.time()
# #     """fits the weights of a two-poissonian distribution to the samples given in hists
# #     parameters:
# #         hists: array of arrays, each one a sample of a two-poissonian distribution
# #             should a sample be empty, the corresponding y, y_err are set to nan
# #         pre_fit: fit result from a fit on all histograms in hists combined
# #             (the expectation values mu1, mu2 are taken and fixed in the fits done here)
# #             (scipy optimize result, from fit_poisson_hist or fit_poisson_hist_rob)
# #         parallel: bool, default False, if parallel computing should be used (better performance, if the mp module works)
# #         remove_nan: bool, default True, sets if nans (from empty lists in hists) should be removed in the output
# #     returns y, y_err
# #         y: a list of weights for the two-poissonian distribution (weight corresponding to mu2, mu2 > mu1)
# #         y_err: errors of the weights
# #     """
#
#     # take variables from prefit
#     fit_mu_1, fit_mu_2, fit_mu_3,_,_,_ = pre_fit['x']
#     [fit_mu_uu, fit_mu_du, fit_mu_dd] = np.sort([fit_mu_1, fit_mu_2, fit_mu_3])
# #     print(fit_mu_uu, fit_mu_du, fit_mu_dd)
#
#     if parallel:
#         args = [(hist, fit_mu_dd, fit_mu_du, fit_mu_uu) for hist in hists]
#         pool = mp.Pool(mp.cpu_count())
#         res_y = pool.starmap(helper_fit_hist_2I, args)
#         time.sleep(0.01)
#         pool.close()
#     else:
#         res_y = []
#         for hist in hists:
#             res_y.append(helper_fit_hist_2I(hist, fit_mu_dd, fit_mu_du, fit_mu_uu))
#
#     # unpack results
#     y_dd, y_du, y_uu = [i[0] for i in res_y], [i[1] for i in res_y], [i[2] for i in res_y]
#     y_err_dd, y_err_du, y_err_uu = [i[3] for i in res_y], [i[4] for i in res_y], [i[5] for i in res_y]
#
#     # print("fit hists:", np.round(time.time()-t1, 3))
#     return y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu

def plot_hist_res_2I(hist, res, maxcounts = 22):
    xx = np.linspace(0, maxcounts, maxcounts+1)
    mu_dd, mu_du, mu_uu = res['x'][0], res['x'][1], res['x'][2]
    p_dd, p_du, p_uu = res['x'][3], res['x'][4], res['x'][5]

    y = p_dd*poisson.pmf(xx, mu_dd) + p_du*poisson.pmf(xx, mu_du) + p_uu*poisson.pmf(xx, mu_uu)
#     y = [(res['x'][2]+res['x'][3])*poisson.pmf(i, res['x'][0]) + (2-res['x'][2]-res['x'][3])*poisson.pmf(i, res['x'][1]) for i in xx]

    plt.hist(hist, bins=range(maxcounts), rwidth=0.8, align='left', density=True)
    plt.plot(xx, y/np.sum(y))
    plt.xlabel("counts")
    plt.ylabel("percantage of occurence")
    plt.show()

def plot_hist_res_2I_old(hist, mus, pops, maxrange=30):
    xx = np.linspace(0, maxrange, maxrange+1)

    y = pops[0]*poisson.pmf(xx, mus[0]) + pops[1]*poisson.pmf(xx, mus[1]) + pops[2]*poisson.pmf(xx, mus[2])

    plt.hist(hist, bins=range(maxrange), rwidth=0.8, align='left', density=True)
    plt.plot(xx, y/np.sum(y))
    plt.show()


############ FIT FUNCTIONS ############
from PyModules.analyse_eios import eios_data

def fit_poisson_from_file(path, prefit = None, ret_prefit = False, as_err = False):
    data = eios_data.read(path)

    n_data = len(data[0])

    hists = [data[0][i]['hists'] for i in range(n_data)]


    if prefit == None:
        MasterHist = np.append([], hists)
        prefit = fit_poisson_hist(MasterHist)
        if ret_prefit:
            return prefit

    res = []
    for i in range(n_data):
        x = data[0][i]['x']
        y, y_err = fit_hist(hists[i], prefit, as_err=as_err)

        res.append([x, y, y_err])
    return res



def fit_poisson_from_files(paths, prefit = None, ret_prefit = False, onlyFirst=False, show_hist=False, as_err = False):
    hists = []
    for path in paths:
        data = eios_data.read(path)
        n_data = len(data[0])

        if onlyFirst:
            hists.append(data[0][0]['hists'])
        else:
            for i in range(n_data):
                hists.append(data[0][i]['hists'])

    if show_hist:
        print(np.shape(hists))
        for hist in hists:
            plt.hist(hist, bins=range(int(np.max(hist))))
            plt.show()

    if prefit == None:
        MasterHist = np.append([], hists)
        prefit = fit_poisson_hist(MasterHist)
        # print(prefit)
        if ret_prefit:
            return prefit

    res = []
    for i, path in enumerate(paths):
        data = eios_data.read(path)
        n_data = len(data[0])

        if onlyFirst:
            x = data[0][0]['x']
            y, y_err = fit_hist(hists[i], prefit, as_err=as_err)
            res.append([x, y, y_err])

        else:
            for j in range(n_data):
                x = data[0][j]['x']
                y, y_err = fit_hist(data[0][j]['hists'], prefit, as_err=as_err)

                res.append([x, y, y_err])
    return res


# from rob
def fit_poisson_hist(hist, lowcount=1., highcount=8., optimizer='iminuit'):
    """fits a two-poissonian distribution (see mLL) to a sample hist, using the scipy minimize function
    optimizer: choose if the optimization should be done with:
        'scipy': like the old function, returns the full scipy optimization result, but errors may not be correct (check the "success" flag), may give problems if values reach their bounds
        'iminuit': more robust optimizer (especially if you want to reuse the errors), returns dictionary of fit results 'x' and their errors 'x_err'
    """
    # if hist is a list of histograms, merge them into one hist
    if len(np.shape(hist)) > 1:
        hist = np.append([],hist)

    if optimizer=='scipy':
        # check mLL for further information on the arguments
        func = lambda args: mLL(np.array(hist), args[0], args[1], args[2])

        # important: first two bounds are not allowed to include zero
        fit = minimize(func, [lowcount, highcount, 0.5], bounds=((0.01, 10), (0.1, 50), (0, 1)), tol=1e-10, method='L-BFGS-B')
        return fit

    elif optimizer=='iminuit':
        def func(mu1, mu2, p_up):
            return mLL(np.array(hist), mu1, mu2, p_up)

        m = iminuit.Minuit(func, mu1 = lowcount, mu2 = highcount, p_up = 0.5,
                    # initial stepsize
                   error_mu1 = 0.01, error_mu2 = 1., error_p_up = 0.05, errordef = 0.5,
                    # bounds
                   limit_mu1 = (0., 200), limit_mu2 = (0., 200),
                   limit_p_up=(0.,1.))
        m.migrad()
        res = {
            'x': [m.values[0], m.values[1], m.values[2]],
            'x_err': [m.errors[0], m.errors[1], m.errors[2]]
        }
        return res

    else:
        raise ValueError("choose optimizer 'scipy' or 'iminuit'")



def fit_poisson_hist_old(fhists, lowcount=0., highcount=4.):
	def llikelihood(data, mu1, mu2, pup, regulate=False):
		if regulate:
			if not (0.<=pup<=1.):
				return -100000.
		pup=np.min([1.,pup])
		pup=np.max([0.,pup])
		return np.sum(np.log(poisson_pdf(data,pup,mu1,mu2)))

	func = lambda args:-llikelihood(fhists,args[0],args[1],args[2])
	fitresult = minimize(func,
						 [lowcount, highcount, 0.5],
						 bounds=((0.,10.), (0.,50.) ,(0.,1.)),
						 tol=1e-10)
	#print('fit1:',fitresult['x'])

	func = lambda args:-llikelihood(fhists,args[0],args[1],args[2], regulate=True)
	fitresult = minimize(func, fitresult['x'], method='bfgs', tol=1e-10)
	#print('fit2:',fitresult['x'])

	return fitresult


import scipy
# helper function for fit_hist_rob, from rob
def helper_fit_hist(hist, fit_mu1, fit_mu2, as_err = False):
    """fits the weights of a two-poissonian distribution (with fixed means fit_mu1, fit_mu2) to the sample given in hists
    returns the weight and its error
    """
    if len(hist) == 0: # make sure the hist is not empty
        return np.nan, np.nan
    else:
        # make a helper function to throw into minuit
        func = lambda p_up: mLL(np.array(hist), fit_mu1, fit_mu2, p_up)
        # make a minuit object
        m = iminuit.Minuit(func, p_up = 0.5, error_p_up = 0.05, errordef = 0.5, limit_p_up=(0.,1.))

        # func = lambda x: mLL(np.array(hist), fit_mu1, fit_mu2, 1/(1+np.exp(x)))
        # m = iminuit.Minuit(func, x = 0, error_x = 0.5, errordef = 0.5)
        # minimize the funciton
        m.migrad()

        p_up = 1/(1+np.exp(m.values[0]))

        # for profile likelihood analysis
        if as_err:
            m.minos()
            err = m.np_merrors()

            # print(m.values['p_up'])
            print(err)
            #
            m.draw_mnprofile('p_up', subtract_min=True)
            plt.show()

        # return the parameter and error (not sure about second summand, taken from original function in eios)
        if as_err:
            # return m.values[0], err
            return p_up, err
        else:
            return m.values[0], np.sqrt(m.errors[0]**2+(0.1/np.sqrt(len(hist)))**2.)


# my function for all hist fits (needs pre fit), from rob
def fit_hist(hists, pre_fit, parallel = True, remove_nan = True, as_err = False):
    t1 = time.time()
    """fits the weights of a two-poissonian distribution to the samples given in hists
    parameters:
        hists: array of arrays, each one a sample of a two-poissonian distribution
            should a sample be empty, the corresponding y, y_err are set to nan
        pre_fit: fit result from a fit on all histograms in hists combined
            (the expectation values mu1, mu2 are taken and fixed in the fits done here)
            (scipy optimize result, from fit_poisson_hist or fit_poisson_hist_rob)
        parallel: bool, default False, if parallel computing should be used (better performance, if the mp module works)
        remove_nan: bool, default True, sets if nans (from empty lists in hists) should be removed in the output
    returns y, y_err
        y: a list of weights for the two-poissonian distribution (weight corresponding to mu2, mu2 > mu1)
        y_err: errors of the weights
    """
    # take variables from prefit
    fit_mu1, fit_mu2, fit_p_up = pre_fit['x']
    # fit_p_up_err = np.sqrt(np.diag(pre_fit['hess_inv'].matmat(np.eye(3)))[-1])

    if parallel:
        args = [(hist, fit_mu1, fit_mu2, as_err) for hist in hists]
        pool = mp.Pool(mp.cpu_count())
        res_y = pool.starmap(helper_fit_hist, args)
        time.sleep(0.01)
        pool.close()
    else:
        res_y = []
        for hist in hists:
            res_y.append(helper_fit_hist(hist, fit_mu1, fit_mu2, as_err))

    # unpack results
    if as_err:
        # print(res_y)
        y = [i[0] for i in res_y]
        y_err_d = [i[1][0][0] for i in res_y]
        # print(y_err_d)
        y_err_u = [i[1][1][0] for i in res_y]
        y_err = np.array([y_err_d, y_err_u])
        return y, [y_err_d, y_err_u]

    else:
        y = [i[0] for i in res_y]
        y_err = [i[1] for i in res_y]

    # remove nans if remove_nan==True
    if remove_nan:
        y = np.array(y)[~np.isnan(y)]
        y_err = np.array(y_err)[~np.isnan(y_err)]

    # print("fit hists:", np.round(time.time()-t1, 3))
    return y, y_err



def fit_hist_old(hists,pre_fit,do_plot=False):
	def stateprob(data, p0,mu1, mu2, dmu1, dmu2):
		ullh=UnbinnedLH(poisson_pdf, data)
		minuit=iminuit.Minuit(ullh, p=p0,
							  error_p=0.001, limit_p=(0.,1.),
							  mu1=mu1, mu2=mu2,
							  fix_mu1=True, fix_mu2=True,
							  pedantic=False, print_level=0)
		minuit.migrad()
		#minuit.hesse()
		pup=minuit.values['p']
		dpup=minuit.errors['p']
		if do_plot:
			hist_plot(data,mu1,mu2,pup)
		return pup, np.sqrt(dpup**2.+(0.1/np.sqrt(len(data)))**2.)#, dprob

	mu1,mu2,pup=pre_fit['x']
	d = np.sqrt(np.diag(pre_fit['hess_inv']))
	dmu1,dmu2,dpup=d

	y=np.empty(len(hists))
	y_err=np.empty_like(y)

	for i, hist in enumerate(hists):
		if len(hist)!=0:
			y[i], y_err[i] = stateprob(hist, pup, mu1, mu2, dmu1, dmu2)
		else:
			y[i] = np.nan
			y_err[i] = np.nan

	if np.isnan(y).max():
		print('delete nan points')
		idx = np.logical_not(np.isnan(y))
		#x=x[idx]
		y=y[idx]
		y_err=y_err[idx]
	return y, y_err


def fit_direct(hists, lowcount=0.2, highcount=6., do_plot=True, pre_fit=None):
    np.seterr(divide='ignore', invalid='ignore')
    if pre_fit == None:
        fhists = np.asarray(hists).flatten()
        pre_fit = fit_poisson_hist(fhists, lowcount, highcount)

    if pre_fit == None and do_plot:
        print(pre_fit['x'])
        mu1,mu2,pup=pre_fit['x']
        hist_plot(fhists,mu1,mu2,pup)

    return fit_hist(hists,pre_fit)


def cal_fidelity(t_pi, print_fit = False):
    nameB,_ = paula.run(xxx)
    nameD,_ = paula.run(xxx)

    dataB
    dataD



def get_goodness_PMP(file, full_flop = True, pre_fit = None, do_plot = False, print_res = False):
    """returns the probability of detecting the bright/dark state after state preparation and manipulation (= MW Flop)
    Two modes:
        full_flop = True:
            in this case, a full MW (or other), the file should contain data of a full MW Flop.
            A sin (with decoherence) is fitted to the flop
            the max/min of the sine (without decoherence) give the probability of detecting the bright/dark state after preparation (without waiting)
        full_flop = False:
            The file should point to an experiment, with only two points taken (=> two histograms)
            the first in the bright state, the second in the dark state (=> after a pi flip, with a prior determined pi time)
            In this case, do_plot has no effect

    parameters:
        file: string, with the path to the file generated by the corresponding experiment
        full_flop: bool, switches between the cases described above
        pre_fit: possibility to give a prefit, to pass prior determined values for mu_1, mu_2 (= means of the two-poissonian-dist.)
        do_plot: plots the data with the fitted sine (only if full_flop = True)
        print_res: prints the propability of detecting the bright/dark stat

    returns (t_pi and error t_pi are set to zero for full_flop = False)
        P_bright, error P_bright, P_dark, error P_dark, t_pi, error t_pi
    """

    # get the poisson fits
    res = fit_poisson_from_file(file, pre_fit)
    xDat = res[0][0]
    yDat = res[0][1]
    eDat = res[0][2]
    L = len(xDat)

    if full_flop:
        # define the weighted chi^2 function that should be minimized
        def func(Up, Low, freq, phi, dec):
            # change variables to get two independant bounds
            A = 0.5 * (Up-Low)
            C = 0.5*(Up+Low)

            chisq = [1/(eDat[i]**2)*(flop(xDat[i], A, freq, phi, dec, C) - yDat[i])**2 for i in range(L)]
            return np.sum(chisq)

        # setup migrad
        m = iminuit.Minuit(func, Up = 0.95, Low = 0.05, freq = 0.05, phi = 0., dec = 0.,
                   error_Up = 0.015, error_Low = 0.015, error_freq = 100., error_phi = 0.1, error_dec = 0.001, errordef = 1,
                   limit_Up = (0,1), limit_Low = (0,1), limit_phi = (0, 2*np.pi), limit_dec = (0, 1))

        # minimize the chi^2 function
        m.migrad()  # run optimiser
        # print(m.values['freq'])

        if do_plot:
            # get the fitted function
            xFlop = np.linspace(0, np.max(res[0][0]), 300)
            yFlop = [flop(i, 0.5*(m.values['Up']-m.values['Low']), m.values['freq'], m.values['phi'], m.values['dec'], 0.5*(m.values['Up']+m.values['Low'])) for i in xFlop]

            # do the plot
            plt.errorbar(xDat, yDat, eDat, fmt = 'bo', capsize = 2)
            plt.plot(xFlop, yFlop)
            plt.xlabel(r'Time ($\mu$s)')
            plt.ylabel(r'$P_{dark}$')
            plt.show()

        if print_res:
            print("Population of bright state: ", np.round(m.values['Up'], 3), "error: ", np.round(m.errors['Up'], 3))
            print("Population of dark state:   ", np.round(1-m.values['Low'], 3), "error: ", np.round(m.errors['Low'], 3))

        # calculate pi time
        t_pi, t_pi_err = freq_to_pi_time( m.values['freq'], m.errors['freq'],  m.values['phi'], m.errors['phi'])
        t_pi, t_pi_err = [0.5/m.values['freq']], [0.5*m.errors['freq']/m.values['freq']**2]
        return m.values['Up'], m.errors['Up'], 1-m.values['Low'], m.errors['Low'], t_pi[0], t_pi_err[0]

    else:
        if print_res:
            print("Population of bright state after 2*t_pi: ", np.round(yDat[1], 3), "error: ", np.round(eDat[1], 3))
            print("Population of dark state after t_pi:   ", np.round(1-yDat[0], 3), "error: ", np.round(eDat[0], 3))

        return yDat[1], eDat[1], 1-yDat[0], eDat[0], 0, 0

def fit_func(func, x, y, y_err, start, limits=None, max_eval=10000, absolute_sigma=False):
    np.seterr(divide='ignore', invalid='ignore')
    try:
        if limits is None:
            popt, pcov = curve_fit(func, x, y, p0=start, sigma=y_err, \
                                   maxfev=max_eval, absolute_sigma=absolute_sigma)
        else:
            popt, pcov = curve_fit(func, x, y, p0=start, sigma=y_err, bounds=limits, \
                                   max_nfev=max_eval, absolute_sigma=absolute_sigma)
        perr = np.sqrt(np.diag(pcov))
        # print(popt)
        # print(pcov)
    except ValueError as e:
        print('ValueError in fit_func: ', e)
        if 0 in y_err:
            print('Y-Error list contains zero values!')
        else:
            print(start)
            print(limits)
        popt = [0.]*len(start)
        perr = popt

    # calc data points of fit
    y_fit = func(x,*popt)
    # reduced chi squared
    chisquared = np.sum(((y-y_fit)/y_err)**2)/(len(y)-len(popt)-1)

    # residual sum of squares
    ss_res = np.sum((y-y_fit)**2)
    # total sum of squares
    ss_tot = np.sum((y-np.mean(y))**2)
    # r-squared
    Rsquared = 1 - (ss_res / ss_tot)

    return popt, perr, Rsquared, chisquared

def fit_linear(x,y,y_err):
    x_max = np.max(x)
    x_min = np.min(x)
    x_width = x_max-x_min

    y_max = np.max(y)
    y_min = np.min(y)
    y_width = y_max-y_min

    m_s = y_width/x_width
    b_s = y_min

    start = [m_s, b_s]
    popt, perr, R2, chi2 = fit_func(func_lin, x, y, y_err, start)
    #popt, perr = round_sig(popt[:], perr[:])
    m, m_err = round_sig([popt[0]],[perr[0]])
    #[m], [m_err] = round_sig([popt[0]],[perr[0]])
    #[b], [b_err] = round_sig([popt[1]],[perr[1]])
    #x0 = [-b/m]
    #x0_err = [abs(-b_err) + abs((b/(m**2))*m_err)]
    #var = [m,b,x0]
    #var_err = [m_err,b_err,x0_err]
    return func_lin, [], start, popt, perr, R2, chi2, m, m_err


def fringe(x, A, x0, omega, g, C):
    delta = x-x0
    P = 1./(1.+(delta/omega)**2) * np.sin(g*np.sqrt(omega**2+delta**2))**2
    return A*P+C

def fit_ramsey(x,y,y_err,width_scale=.1):
    x_max = np.max(x)
    x_min = np.min(x)
    x_width = x_max-x_min
    dx = x[1]-x[0]

    f_min = .5/x_width
    f_max = .5/dx

    y_max = np.max(y)
    y_min = np.min(y)
    y_width = (y_max-y_min)

    # Offset
    C = y_min
    A = 3.*y_width

    # peak position
    peaks = np.array(find_peaks(y,width=5,strict=False)) #5, 7
    idx = np.argsort(y[peaks])
    idx = idx[::-1]
    peaks = peaks[idx]
    x_high = (np.mean(np.diff(x[peaks])))

    #x0 = x[peaks[0]]
    x0 = x_min+x_width/2.
    g = 2.5/x_high

    width = np.abs(np.max(x)-np.min(x))*width_scale
    start = [A,x0,width,g,C]
    popt, perr, R2, chi2 = fit_func(fringe, x, y, y_err, start)
    #popt, perr = round_sig(popt[:], perr[:])

    freq, freq_err = round_sig([popt[1]],[perr[1]])
    #freq, freq_err = [popt[1]], [perr[1]]
    print(start)
    print(popt)
    return fringe, peaks, start, popt, perr, R2, chi2, freq, freq_err

def fit_single_abs_pos(func, x, y, y_err, N=1, peak_window=10):
    # peak position
    peaks = np.array(find_peaks(y,width=peak_window,strict=False)) #5, 7
    if len(peaks)<1:
        return None, [], [], [], [], 0., 0., [], []
    if N is None:
        N = len(peaks)
    idx = np.argsort(y[peaks])
    idx = idx[::-1]
    peaks = peaks[idx]

    idx_min = np.argmin(y)
    idx_max = peaks[0]
    y_min = y[idx_min]
    y_max = y[idx_max]
    C = y_max
    x0 = x[idx_max]
    A = (y_min-y_max)/abs(x[idx_min]-x0)

    start = [C,A,x0]

    popt, perr, R2, chi2 = fit_func(func, x, y, y_err, start)
    pos, pos_err = round_sig(popt[-N:],perr[-N:])

    return func, peaks, start, popt, perr, R2, chi2, pos, pos_err

def fit_multi_freq(func,x,y,y_err,N=None, width_scale=.2, peak_window=5):
    x_max = np.max(x)
    x_min = np.min(x)
    x_width = (x_max-x_min)
    x_width_min = x[1]-x[0]
    sigma_lower = x_width_min/2.
    sigma_upper = 4.*x_width
    x0_lower = x_min-x_width/2.
    x0_upper = x_max+x_width/2.

    y_max = np.max(y)
    y_min = np.min(y)
    y_width = (y_max-y_min)
    A_lower = -4.*y_width
    A_upper = +4.*y_width

    # Offset
    C = y_min
    C_lim = [C-y_width, C+y_width]
    C_lower = np.min(C_lim)
    C_upper = np.max(C_lim)


    # peak position
    peaks = np.array(find_peaks(y,width=peak_window,strict=False)) #5, 7
    if len(peaks)<1:
        return None, [], [], [], [], 0., 0., [], []
    if N is None:
        N = len(peaks)
    idx = np.argsort(y[peaks])
    idx = idx[::-1]
    peaks = peaks[idx]
    a = []; x0 = []
    for i,p in zip(range(N),peaks):
        a.append(y[p]-C)
        x0.append(x[p])
    # width = 20% of scan range, bold
    width = x_width*width_scale/N
    sigma = [width]*N

    # func(x, C, *[A], *[SIGMA], *[X0])
    start = [C,*a,*sigma,*x0]
    #print(start)
    limits = ([C_lower]+([A_lower]*N)+([sigma_lower]*N)+([x0_lower]*N) , [C_upper]+([A_upper]*N)+([sigma_upper]*N)+([x0_upper]*N))
    #print(limits)

    popt, perr, R2, chi2 = fit_func(func, x, y, y_err, start, limits=limits)
    #popt, perr = round_sig(popt[:], perr[:])

    freq, freq_err = round_sig(popt[-N:],perr[-N:])
    #freq, freq_err = popt[-N:], perr[-N:]

    return func, peaks, start, popt, perr, R2, chi2, freq, freq_err

def freq_to_pi_time(freq, freq_err, phase, phase_error):
    T = 1./freq
    T_err = abs(-1./(freq**2))*freq_err

    phase = math.fmod(phase,2.*np.pi)

    t_shift = T*phase/(2.*np.pi)
    t_flip = T/2.-t_shift
    t_flip_err = abs(.5-phase/(2.*np.pi))*T_err + abs(T/(2.*np.pi))*phase_error
    #print(phase_error)
    #print(phase/(2.*np.pi))
    #print(T,T_err)
    #print(freq,freq_err)
    #print(phase,t_shift)
    #print(t_flip,t_flip_err)

    t_pi = [t_flip]
    t_pi_err = [t_flip_err]
    t_pi, t_pi_err = round_sig(t_pi[:], t_pi_err[:])

    return t_pi, t_pi_err

def fit_time(t, y, y_err, phi=0., gamma=0.):
    # phi = 3*np.pi/2
    # flop(t, A, freq, phi, gamma, C)

    t_max = np.max(t)
    t_min = np.min(t)
    t_width = t_max-t_min
    dt = t[1]-t[0]

    f_min = .5/t_width
    f_max = .5/dt

    y_max = np.max(y)
    y_min = np.min(y)
    y_width = (y_max-y_min)
    C = np.mean(y)
    A = y_width/2.

    peaks = find_peaks(y,width=7,strict=True)
    if len(peaks)<2:
        t_2pi = t_max
    else:
        t_2pi = (np.mean(np.diff(t[peaks])))
    freq = 1./t_2pi
    start = [A,freq,phi,gamma,C]
    #print(len(peaks),peaks,t_2pi)

    C_lim = [C-y_width, C+y_width]
    C_min = np.min(C_lim)
    C_max = np.max(C_lim)

    limits=([0.,f_min,-1.5*np.pi,0.,C_min],[y_width,f_max,1.5*np.pi,t_width,C_max])
    popt, perr, R2, chi2 = fit_func(flop, t, y, y_err, start, limits=limits)
    #popt, perr = round_sig(popt[:], perr[:])

    t_pi, t_pi_err = freq_to_pi_time(popt[1], perr[1], popt[2], perr[2])
    return flop, peaks, start, popt, perr, R2, chi2, t_pi, t_pi_err

def fit_phase(t, y, y_err, phi=0., f=1., f_min=1., f_max=np.pi):
    # phi = 3*np.pi/2
    # flop(t, A, freq, phi, gamma, C)

    t_max = np.max(t)
    t_min = np.min(t)
    t_width = t_max-t_min
    dt = t[1]-t[0]

    y_max = np.max(y)
    y_min = np.min(y)
    y_width = (y_max-y_min)
    C = np.mean(y)
    A = y_width/2.

    peaks = find_peaks(y,width=7,strict=True)
    #if len(peaks)<2:
    #    t_2pi = t_max
    #else:
    #    t_2pi = (np.mean(np.diff(t[peaks])))
    #freq = 1./t_2pi

    start = [A,f,phi,C]
    #print(len(peaks),peaks,t_2pi)

    C_lim = [C-y_width, C+y_width]
    C_min = np.min(C_lim)
    C_max = np.max(C_lim)

    limits=([0.,f_min,-1.5*np.pi,C_min],[y_width,f_max,1.5*np.pi,C_max])
    popt, perr, R2, chi2 = fit_func(phase, t, y, y_err, start, limits=limits)
    #popt, perr = round_sig(popt[:], perr[:])

    t_pi, t_pi_err = freq_to_pi_time(popt[1], perr[1], popt[2], perr[2])
    return phase, peaks, start, popt, perr, R2, chi2, t_pi, t_pi_err

def fit_parameter(func, data, invert=False, verbose=False):
    x,y,y_err = unpack_sorted(data)

    # peak or dip
    if invert:
        y = -y
    # call function (estimate+fit)
    func_model, peaks, start, popt, perr, R2, chi2, var, var_err = func(x,y,y_err)

    if verbose:
        for v,err in zip(var,var_err):
            n = significant_digit(err)
            s = '%%.%if'%max(n,0)
            print('\t%s(%i) (%6.2f%%)' % (s%v,(err*10**n),R2*100))
            #print('\t%s +- %s (%6.2f%%)' % (s%v,s%err,R2*100))

    return x, y, y_err, func_model, peaks, start, popt, perr, R2, chi2, var, var_err

def plot_data(ax, data, label):
    for data_cntr, lbl in zip(data,label):
        x = data_cntr['x']
        y = data_cntr['y']
        y_err = data_cntr['error']
        ax.errorbar(x, y, y_err, marker = 'o', ls='',fmt='', markersize=7.5, lw=1., capsize=.0, label=lbl)
        #ax.errorbar(x, y, y_err, marker = 'o', markersize=3., color='C0', lw=1, ls='',fmt='',capsize=2, label=lbl)

def plot_fit(x,y,y_err,peaks,func,start,popt,perr,\
             var,var_err,gof,chi2,invert,\
             plot_start=False, plot_patch=False, \
             plot_residuals=False, plot_peaks=True,
             plot_label=[None, None], lbl=[None, None]):

    # Text box with fit results and errors
    arg_names = inspect.getfullargspec(func).args[1:]
    #txt = r'   $R^2$ = %6.2f%%'%(gof*100) + '\n' + r'   $\chi^2$ = %6.2f'%(chi2) + '\n'
    txt = r'   $R^2$ = %.2f%%'%(gof*100) + '\n'
    for m,v,e in zip(arg_names, popt, perr):
        n = max(significant_digit(e),0)
        #s_v = '%%%i.%if'%(6+n,n)
        #s_e = '%%%i.%if'%(1+n,n)
        #txt += '%5s'%m + ' '+r'= ' + s_v%v + r' $\pm$ ' + s_e%e + '\n'
        s_v = '%%.%if'%(n)
        txt += '%5s'%m + ' '+r'= ' + s_v%v
        if np.isfinite(e):
            txt += '(%i)\n'%(e*10**n)
        else:
            txt += '(inf)\n'
    txt = txt[:-1]

    X = np.linspace(min(x),max(x),1000)
    Y_fit = func(X,*popt)
    y_fit = func(x,*popt)
    if invert:
        y = -y
        Y_fit = -Y_fit
        y_fit = -y_fit

    # Plot residuals
    if plot_residuals: #figsize=(8,6)
        dy = y-y_fit
        f, (ax,ax_res) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]}, sharex=True)
        ax_res.errorbar(x, dy, y_err, color='Navy', ls='',fmt='', marker='o', markersize=7.5, lw=1., capsize=.0)
        ax_res.axhline(y=0., ls='-', c='Orange', alpha=.5, lw=3)
        ax_res.set_ylabel('Res.')
        if lbl[0] is not None:
            ax_res.set_xlabel(lbl[0])
        axs = (ax, ax_res)
    else:
        f, ax = plt.subplots()
        axs = ax

    # Plot fit model with results
    ax.plot(X,Y_fit,'r-',label=plot_label[1], lw=3., color='Orange', alpha=.5)
    # add label
    if lbl[0] is not None and plot_residuals == False:
        ax.set_xlabel(lbl[0])
    if lbl[1] is not None:
        ax.set_ylabel(lbl[1])

    # Plot start parameter of fit
    if plot_start:
        Y_srt = func(X,*start)
        if invert:
            Y_srt = -Y_srt
        ax.plot(X,Y_srt,'--')

    # Plot data points with errorbars
    ax.errorbar(x, y, y_err, \
                color='Navy', ls='',fmt='', \
                marker='o', markersize=7.5, lw=1., capsize=.0,\
                label=plot_label[0])
    if plot_label[0] is not None:
        ax.legend(loc=0)

    # Plot peaks
    if plot_peaks:
        peak_off = (np.max(y)-np.min(y))*.15
        peak_marker = 'v'
        if invert:
            peak_off = -peak_off
            peak_marker = '^'
        ax.plot(x[peaks],y[peaks]+peak_off,peak_marker)

    # Add uncertainty patches
    if plot_patch and len(popt) == 2:
        # 2-parameter fit model (e.g. linear)
        y1 = func(X,*(popt-perr))
        y2 = func(X,*(popt+perr))
        ax.fill_between(X, y1, y2, color='xkcd:cyan', alpha=.5)
    elif (len(popt) == 3) or (len(popt) == 4) or (len(popt) == 5):
        # 4- (frequency e.g gauss), 5-parameter fit model (flop)
        for v,err in zip(var,var_err):
            ax.axvspan(v-err, v+err, label='uncertainty', facecolor='Orange', alpha=.5, edgecolor='None')
            ax.axvline(x=v, ls='-', label='opt. value', c='Orange', lw=3.)

    # set limits x-axes
    spacing = (x[1]-x[0])/2.
    lower = min(x)-spacing
    upper = max(x)+spacing
    ax.set_xlim(lower, upper)

    # Append text box to plot
    plt.rcParams['font.family'] = 'monospace'
    ax.text(1.02, .18, txt, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
    #ax.get_xaxis().get_major_formatter().set_useOffset(False)
    import datetime
    fname='./data/scans/'+str(datetime.datetime.now())[:22].replace(' ','_').replace(':','-').replace('.','-')+'.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.75)
    return f, ax

# Plot the histogram fit
def hist_plot(fhists,mu1,mu2,pup):
	x = np.arange(15)
	cnt_bins = np.arange(-0.5, 15.5, 1)
	plt.figure('\lambda_1 = %.2e \lambda_1 = %.2e P = %f' % (mu1,mu2,pup))
	plt.plot(x,poisson_pdf(x, pup, mu1, mu2))
	plt.hist(fhists, bins=cnt_bins, density=True, rwidth=0.85)
	plt.plot(x,((pup*poisson.pmf(x, mu2))))
	plt.plot(x,(((1.-pup)*poisson.pmf(x, mu1))))

	plt.xlabel('Number of photons')
	plt.ylabel(r'$N_\mathrm{\#} /\left(N_{\mathrm{scan}}\cdot N_{\mathrm{exp}}\right)$')
	plt.xlim(-1,15)
	plt.tight_layout(pad=0.,rect=[0.01, 0.01, 0.99, 0.99])
	plt.show()
	# plt.savefig('tmp.svg')
