import numpy as np
import math
import iminuit

# included from rob
import multiprocessing as mp
import time


import matplotlib.pyplot as plt

from scipy.stats import poisson
from scipy.optimize import minimize
# from scipy.optimize import curve_fit

from PyModules.analyse_eios import eios_data

################### 1 Ion  #####################
def mLL(data, mu1, mu2, p_up, remove_inf=False):
    """returns the negative log likelihood functin of a two-poissonian distribution with:
        means mu1 and mu2
        weights (1-p_up) and fit_p_up
        data: an array filled with random numbers from the two-poissonian distribution
    """
    # make sure that p_up \in [0, 1]
    if not (0<= p_up <= 1):
        p_up = np.max([p_up, 0])
        p_up = np.min([p_up, 1])

    # print(len(data))
    list = np.log((1-p_up)*poisson.pmf(data, mu1) + p_up*poisson.pmf(data, mu2))

    # remove inf's
    if True:
        l = np.isinf(list)
        list = np.array(list)[~np.isinf(list)]
        # print(np.sum(l))
    # print(np.max(np.abs(list)))
    return(-np.sum(list) + np.sum(l)*np.abs(np.sum(list)))

def fit_poisson_from_file(path, prefit = None, doplot = False):
    data = eios_data.read(path)

    n_data = len(data[0])

    hists = [data[0][i]['hists'] for i in range(n_data)]

    if prefit == None:
        MasterHist = np.append([], hists)
        prefit = fit_poisson_hist(MasterHist)

    res = []
    for i in range(n_data):
        x = data[0][i]['x']
        y, y_err = fit_hist(hists[i], prefit)

        res.append([x, y, y_err])
        
    if doplot:
        for r in res:
            plt.errorbar(res[0][0], res[0][1], res[0][2], fmt='x')
            
        plt.show()
    return res


# from rob
def fit_poisson_hist(hist, lowcount=1., highcount=8., optimizer='iminuit', limit = 50):
    """fits a two-poissonian distribution (see mLL) to a sample hist, using the scipy minimize function
    optimizer: choose if the optimization should be done with:
        'scipy': like the old function, returns the full scipy optimization result, but errors may not be correct (check the "success" flag), may give problems if values reach their bounds
        'iminuit': more robust optimizer (especially if you want to reuse the errors), returns dictionary of fit results 'x' and their errors 'x_err'
    """
    # if hist is a list of histograms, merge them into one hist
    if len(np.shape(hist)) > 1:
        hist = np.append([],hist)

    # print(len(hist))

    if optimizer=='scipy':
        # check mLL for further information on the arguments
        func = lambda args: mLL(np.array(hist), args[0], args[1], args[2])

        # important: first two bounds are not allowed to include zero
        fit = minimize(func, [lowcount, highcount, 0.5], bounds=((0.01, limit), (0.1, limit), (0, 1)), tol=1e-10, method='L-BFGS-B')
        return fit

    elif optimizer=='iminuit':
        def func(mu1, mu2, p_up):
            return mLL(np.array(hist), mu1, mu2, p_up)

        m = iminuit.Minuit(func, mu1 = lowcount, mu2 = highcount, p_up = 0.5,
        # m = iminuit.Minuit(func, mu1 = 1., mu2 = 8., p_up = 0.5,
                    # initial stepsize
                   error_mu1 = 0.01, error_mu2 = 1., error_p_up = 0.05, errordef = 0.5,
                    # bounds
                   limit_mu1 = (0., limit), limit_mu2 = (0., limit), limit_p_up=(0.,1.))
        fmin,_ = m.migrad()

        # print(fmin['is_valid'])
        # print(m.values)
        res = {
            'x': [m.values[0], m.values[1], m.values[2]],
            'x_err': [m.errors[0], m.errors[1], m.errors[2]]
        }
        return res

    else:
        raise ValueError("choose optimizer 'scipy' or 'iminuit'")
        
        
def helper_fit_hist(hist, fit_mu1, fit_mu2):
    """fits the weights of a two-poissonian distribution (with fixed means fit_mu1, fit_mu2) to the sample given in hists
    returns the weight and its error
    """
    if len(hist) == 0: # make sure the hist is not empty
        print("hist empty")
        return np.nan, np.nan
    else:
        # make a helper function to throw into minuit
        func = lambda p_up: mLL(np.array(hist), fit_mu1, fit_mu2, p_up)
        # make a minuit object
        m = iminuit.Minuit(func, p_up = 0.5, error_p_up = 0.05, errordef = 0.5, limit_p_up=(0.,1.))
        # minimize the funciton
        fmin,_ = m.migrad()
        # print(fmin['is_valid'], m.values[0])
        # return the parameter and error (not sure about second summand, taken from original function in eios)
        return m.values[0], np.sqrt(m.errors[0]**2+(0.1/np.sqrt(len(hist)))**2.)
    

# my function for all hist fits (needs pre fit), from rob
def fit_hist(hists, pre_fit, parallel = True, remove_nan = True):
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
        args = [(hist, fit_mu1, fit_mu2) for hist in hists]
        pool = mp.Pool(mp.cpu_count())
        res_y = pool.starmap(helper_fit_hist, args)
        time.sleep(0.01)
        pool.close()
    else:
        res_y = []
        for hist in hists:
            res_y.append(helper_fit_hist(hist, fit_mu1, fit_mu2))

    # unpack results
    y = [i[0] for i in res_y]
    y_err = [i[1] for i in res_y]

    # remove nans if remove_nan==True
    if remove_nan:
        y = np.array(y)[~np.isnan(y)]
        y_err = np.array(y_err)[~np.isnan(y_err)]

    # print("fit hists:", np.round(time.time()-t1, 3))
    return y, y_err




################### 2 Ions #####################
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


def fit_poisson_hist_2I(hist, lowcount=1., highcount=16., limit=50):
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
               limit_mu_dd = (0., limit), limit_mu_du = (0., limit), limit_mu_uu = (0., limit),
               limit_p_dd=(0.,1.), limit_p_du=(0.,1.), limit_p_uu=(0.,1.))
    m.migrad()

    # make sure the probabilities add up to 1
    norm = m.values[3] + m.values[4] + m.values[5]

    res = {
        'x': [m.values[0], m.values[1], m.values[2], m.values[3]/norm, m.values[4]/norm, m.values[5]/norm],
        'x_err': [m.errors[0], m.errors[1], m.errors[2], m.errors[3]/norm, m.errors[4]/norm, m.errors[5]/norm]
    }
    return res


def helper_fit_hist_2I(hist, fit_mu_dd, fit_mu_du, fit_mu_uu):
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

        # make sure the probabilities add up to 1
        norm = p_dd + p_du + p_uu
        if np.abs(1-norm) > 0.01:
            print("probabilities not normalized!!!")

        # calculate the errors of p_dd, p_du, p_uu
        err0 = np.sqrt((m.errors[0]*(1-m.values[1]))**2 + (m.values[0]*m.errors[1])**2)
        err1 = m.errors[0]
        err2 = np.sqrt((m.errors[0]*m.values[1])**2 + (m.values[0]*m.errors[1])**2)

        # return the parameter and error
        return p_dd, p_du, p_uu, err0, err1, err2


# my function for all hist fits (needs pre fit), from rob
def fit_hist_2I(hists, pre_fit, parallel = True):
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
        args = [(hist, fit_mu_dd, fit_mu_du, fit_mu_uu) for hist in hists]
        pool = mp.Pool(mp.cpu_count())
        res_y = pool.starmap(helper_fit_hist_2I, args)
        time.sleep(0.01)
        pool.close()
    else:
        res_y = []
        for hist in hists:
            res_y.append(helper_fit_hist_2I(hist, fit_mu_dd, fit_mu_du, fit_mu_uu))

    # unpack results
    y_dd, y_du, y_uu = [i[0] for i in res_y], [i[1] for i in res_y], [i[2] for i in res_y]
    y_err_dd, y_err_du, y_err_uu = [i[3] for i in res_y], [i[4] for i in res_y], [i[5] for i in res_y]

    # print("fit hists:", np.round(time.time()-t1, 3))
    return y_dd, y_du, y_uu, y_err_dd, y_err_du, y_err_uu

def plot_hist_res_2I(hist, mus, pops, maxrange=30):
    xx = np.linspace(0, maxrange, maxrange+1)

    y = pops[0]*poisson.pmf(xx, mus[0]) + pops[1]*poisson.pmf(xx, mus[1]) + pops[2]*poisson.pmf(xx, mus[2])

    plt.hist(hist, bins=range(maxrange), rwidth=0.8, align='left', density=True)
    plt.plot(xx, y/np.sum(y))
    plt.show()