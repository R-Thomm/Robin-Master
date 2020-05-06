import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import moment

import sys
sys.path.insert(0, './../')

from PyModules.ipc_client import IPC_client
from PyModules.analyse_eios.eios_data import save, load
from PyModules.analyse_eios.eios_data import current_result_dir

def parse(ret):
    lines = ret.split('\n')
    head = lines[:3]
    body = lines[3:]
    h = head[1].split('\t')
    N = float(h[0])
    R = float(h[1])
    n = []; t = []
    b = []; y = []
    for p in body:
        d = p.split('\t')
        if len(d)>3:
            n.append(float(d[0]))
            t.append(float(d[1]))
            b.append(float(d[2]))
            y.append(float(d[3]))
    return N,R,np.array(n),np.array(t),np.array(b),np.array(y)

def pcm_analyse(y):
    mean = np.mean(y)
    var = moment(y, moment=2)
    std = np.sqrt(var)
    skew = moment(y, moment=3)/(std**3)
    kurtosis = moment(y, moment=4)/(std**4)
    #y_ref = y[-1]
    y_ref = (y[0]+y[-1])/2.
    fom = np.sum(y-y_ref)/np.sum(y)
    return mean,std,skew,kurtosis,fom

def pcm_capture(ipc_client,start=6,stop=-3):
    ret = ipc_client.receive()
    N,R,n,t,bins,nrm = parse(ret)
    return N,R,n,t,bins,nrm

def pcm_save(path,storage):
    profile = storage['profile']
    shim = storage['shim']
    time_str = storage['time_start']
    name = 'data_pcm_%s_%s_%s.json'%(profile,shim,time_str)
    filepath = '%s/%s'%(path,name)
    save(filepath,storage)
    return filepath

def pcm_open(name):
    storage = load(name)
    return storage

def pcm_plot_hists(storage):
    # histograms
    x = np.array(storage['x'])
    hists = np.array(storage['hists'])
    hists_err = np.array(storage['hists_err'])

    hists = np.array([hists[0,:],hists[hists.shape[0]//2,:],hists[-1,:]])
    hists_err = np.array([hists_err[0,:],hists_err[hists_err.shape[0]//2,:],hists_err[-1,:]])

    dx = (x[1]-x[0])*.75
    
    fig, axs = plt.subplots(hists.shape[0], 1)
    for ax,h,err in zip(axs,hists,hists_err):
        ax.bar(x, h, width=dx, yerr=err, align='center', alpha=1.)
    return fig, axs

def pcm_plot_stats(storage):
    # statistics
    voltage = np.array(storage['parameter'])
    stats = np.array(storage['stats'])
    stats_err = np.array(storage['stats_err'])

    lbl = ['mean','std. dev.','skewness','kurtosis','fom']
    fig, axs = plt.subplots(stats.shape[1], 1, sharex=True)
    for i,ax in enumerate(axs):
        ax.errorbar(voltage, stats[:,i], yerr=stats_err[:,i], fmt='--', marker='x', label=lbl[i])
        ax.legend()
    axs[-1].axhline(y=0,linewidth=.5, color='k')
    return fig, axs

class PCM(object):
    def __init__(self, results_path, server_address='/tmp/pcm-socket'):
        self.results_path = results_path
        self.client = IPC_client(server_address)
        
    def get_raw(self):
        N,R,n,t,bins,nrm = pcm_capture(self.client)
        return N,R,n,t,bins,nrm

    def analyse(self, bins, start=6, stop=-3):
        mean,std,skew,kurtosis,fom = pcm_analyse(bins[start:stop])
        return mean,std,skew,kurtosis,fom

    def get_analyse(self, start=6, stop=-3, verbose=False):
        N,R,n,t,bins,nrm = self.get_raw()
        mean,std,skew,kurtosis,fom = self.analyse(bins, start=start, stop=stop)
        arr = np.reshape([mean,std,skew,kurtosis,fom],(1,5))
        if verbose:
            print('N=%i, rate %.3f kHz'%(N,R*1e-3))
            print('\tMean %8.2e, Std %8.2e, Skw %8.2e, Krt %8.2e'%(mean,std,skew,kurtosis))
        return n,t,bins,nrm,arr,R

    def pack(self, time_start, time_stop, profile_name, shim_name, init_value, parameter, t, hists, hists_err, stats, stats_err, rate):
        storage = { 'time_start':time_start,
                    'time_stop':time_stop,
                    'profile':profile_name, 
                    'shim':shim_name,
                    'shim_init':init_value,
                    'parameter':parameter.tolist(), 
                    'x':t.tolist(), 
                    'hists':hists, 'hists_err':hists_err, 
                    'stats':stats, 'stats_err':stats_err,
                    'rate':rate}
        return storage

    def unpack(self, storage):
        parameter = np.array(storage['parameter'])
        x = np.array(storage['x'])
        dx = (x[1]-x[0])*.75
        hists = np.array(storage['hists'])
        hists_err = np.array(storage['hists_err'])
        stats = np.array(storage['stats'])
        stats_err = np.array(storage['stats_err'])
        rate = []
        if 'rate' in storage:
            rate = storage['rate']
        rate = np.array(rate)
        return parameter, x, dx, hists, hists_err, stats, stats_err, rate

    def save(self, storage, file_path=''):
        if not file_path:
            path = current_result_dir(local_dir=self.results_path,time_str=storage['time_start'])
            file_path = path+'/PCM'
        return pcm_save(file_path, storage)

    def load(self, file_path):
        return pcm_open(file_path)

    def plot_hists(self, storage):
        fig, axs = pcm_plot_hists(storage)
        plt.show()

    def plot_stats(self, storage):
        fig, axs = pcm_plot_stats(storage)
        plt.show()

if __name__ == "__main__":
    pcm = PCM()
    #N,R,n,t,bins,nrm = pcm.get_raw()
    #mean,std,skew,kurtosis,fom = pcm.analyse(bins)
    n,t,bins,nrm,arr = pcm.get_analyse()


