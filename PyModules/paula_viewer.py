#!/usr/bin/env python3

#flo
from scipy.optimize import curve_fit
import random
####


import datetime
import numpy as np
import time
import math
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from IPython.display import display, HTML

from PyModules.epos_viewer import EPOS_V
import PyModules.analyse_eios.eios_data as eios_file
from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save
from PyModules.analyse_eios.eios_analyse import significant_digit, round_sig, plot_fit
from PyModules.analyse_eios.eios_analyse import fit_func, func_decay_exponential, func_decay_reciprocal, gauss, sincSquare, fit_linear, func_lin
from PyModules.utilities import do_async, integer_hill_climb, wait

class PAULA_V(EPOS_V):

    def __init__(self, results_path='/Volumes/Paula_Data', \
                    cache_path='./data/', \
                    log_file='/Volumes/Paula_EIOS/Notebooks/data/log.txt', log_file_org='./data/log.txt', \
                    wvf_folder='/Volumes/Paula_EIOS/UserData', \
                    wvf_db_file='/Volumes/Paula_EIOS/waveform_db.json', \
                    gof_thres=0.7, dark=False):
        super().__init__(cache_path=cache_path,log_file=log_file, log_file_org=log_file_org, do_plot=True, \
                            gof_thres=gof_thres, wvf_folder=wvf_folder, wvf_db_file=wvf_db_file)

        self.results_path = results_path

        sns.set()
        if dark==True:
            COLOR = 'white'
            BCKCOLOR = 'white'
            GRIDCOLOR = 'grey'
        else:
            COLOR = 'black'
            BCKCOLOR = 'white'
            GRIDCOLOR = 'black'
        mpl.rcParams['text.color'] = COLOR
        mpl.rcParams['axes.labelcolor'] = COLOR
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['xtick.color'] = COLOR
        mpl.rcParams['ytick.color'] = COLOR
        mpl.rcParams['axes.facecolor'] = BCKCOLOR
        mpl.rcParams['grid.color'] = GRIDCOLOR
        mpl.rcParams['grid.linestyle'] = '-'
        mpl.rcParams['grid.linewidth'] = 0.5
        mpl.rcParams['errorbar.capsize'] = 2.0
        mpl.rcParams['lines.markersize'] = 4.0
        mpl.rcParams['patch.facecolor'] = BCKCOLOR
        mpl.rcParams['patch.force_edgecolor'] = True
        mpl.rcParams["savefig.dpi"] = 125
        sns.set()

    def session_replay_fit_sim_freq_iter(self,session_ids):
        data_all = []
        data_list = []
        t_heat_list = []
        data_concencated = self.session_replay(session_ids)
        for j in range(len(data_concencated)):
            try:
                data_all.append(data_concencated[:data_concencated.index("+")])
                data_concencated = data_concencated[data_concencated.index("+")+1:]
            except:
                break

        for i in range(len(data_all)):
            data = self.session_replay(data_all[i][:19])
            xb=data[0]['x']
            yb=data[0]['y']
            yerrb=data[0]['error']
            xr=data[1]['x']
            yr=data[1]['y']
            yerrr=data[1]['error']
            data = [[xb,yb,yerrb],[xr,yr,yerrr]]
            data_list.append(data)
            t_heat_list.append(data_all[i][20:])
        return data_list, t_heat_list

    def fit_simultaneously_freq_iter(self, data_list, t_heat_list, ret_list=None, plot_together=True, set_dur=False):
        nbar_list = []
        nbar_err_list = []
        t_heat_list_new = []
        for i in range(len(data_list)):
            data = [data_list[i][0], data_list[i][1]]
            nbar,nbar_err = self.fit_simultaneously_freq(data,ret_list=ret_list,set_dur=set_dur)
            nbar_list.append(nbar)
            nbar_err_list.append(nbar_err)
            t_heat_list_new.append(float(t_heat_list[i]))
            try:
                x = np.array(t_heat_list_new)
                y = np.array(nbar_list)
                yerr = np.array(nbar_err_list)
                data = [x,y,yerr]
                self.do_plot = True
                ret_lin = self.single_fit_func(data, self.fit_linear, invert=False, add_contrast=False, xlabel='t_heat in ms', ylabel='average n', give_all_para=True)
            except:
                print('no fit')

    def session_replay_fit_simultaneously_freq_iter(self,session_ids,set_dur=False):
        data_list,t_heat_list = self.session_replay_fit_sim_freq_iter(session_ids)
        self.fit_simultaneously_freq_iter(data_list, t_heat_list, ret_list=None, plot_together=True,set_dur=set_dur)


    def session_replay_meta(self,sid, experiment=None, do_plot=True, invert=False):
        #reorder the session_id to get the IP when the experiment starts
        #time_start =

        ret_list, data_list = self.session_replay(sid, do_plot=do_plot)

        if experiment == None:
            return

        x = []
        y = []
        yerr = []
        start_IP_history = sid[:10]+' '+sid[11:13]+':'+sid[14:16]+':'+sid[17:19]
        #arbitrary date
        end_IP_history = '3'+start_IP_history[1:]

        if experiment == 'mw_Ramsey_coherence':
            func = func_decay_exponential
            print(func)
            for i in range(len(ret_list)):
                #Das hier macht natuerlich probleme sobald die Variable zwischendurch auf 0 gestellt wird und die reinfolge zu aender ist dann auch scheisse weil man dann nicht einfach das letzte als erstes nehmen kann und minus eine Sekunde bei der start_IP macht dann bei 00 ein Problem
                x.append(self.plt_longterm(ip_l=['t_ramsey'], rng=[start_IP_history,end_IP_history], rel=1,verbose=False)[0][1][i][1])
                y.append(ret_list[i][3][0])
                yerr.append(ret_list[i][4][0])


            x = np.sort(x)
            norm = y[0]
            y = y/norm
            yerr = yerr/norm
            data = [np.array(x),np.array(y),np.array(yerr)]
            guess = [100,1]

        popt, perr, R2, chi2 = self.fit_data(func, data, guess)
        '''self.single_fit_func([np.array(x), np.array(y), np.array(yerr)], func, invert=invert, \
                                                       give_all_para=True, name_var='', \
                                                       xlabel=None, ylabel=None, add_contrast=True, \
                                                       text=None, title='title', set_fit=True)'''
