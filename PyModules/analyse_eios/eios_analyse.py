import numpy as np
import math
import iminuit
import inspect

import matplotlib.pyplot as plt

from scipy.stats import poisson
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

############ FUNCTIONS ############

def func_decay_exponential(x, tau, exponent):
    return np.exp(-(x/tau)**exponent)

def func_decay_reciprocal(x, bndwdth):
    return (1./x)+bndwdth*1e-6

def poisson_pdf(x, p, mu1, mu2):
    return (1.-p)*poisson.pmf(x, mu1)+p*poisson.pmf(x, mu2)

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

############ FIT FUNCTIONS ############

def fit_poisson_hist(fhists, lowcount=0., highcount=4.):
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

def fit_hist(hists,pre_fit,do_plot=False):
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
	plt.hist(fhists, bins=cnt_bins, density=True)
	plt.plot(x,((pup*poisson.pmf(x, mu2))))
	plt.plot(x,(((1.-pup)*poisson.pmf(x, mu1))))

	plt.xlabel('Number of photons')
	plt.ylabel(r'$N_\mathrm{\#} /\left(N_{\mathrm{scan}}\cdot N_{\mathrm{exp}}\right)$')
	plt.xlim(-1,15)
	plt.tight_layout(pad=0.,rect=[0.01, 0.01, 0.99, 0.99])
	plt.show()
	# plt.savefig('tmp.svg')
