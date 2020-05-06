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

class EURO_V(EPOS_V):

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
