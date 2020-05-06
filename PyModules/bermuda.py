#!/usr/bin/env python3

import datetime
import numpy as np
import time
import math

from threading import Thread
from PyModules.wavemeter.pid_wrapper import pid_container

from PyModules.epos import EPOS
from PyModules.utilities import do_async, integer_hill_climb, wait
from PyModules.analyse_eios.eios_analyse import significant_digit, round_sig
from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

class BERMUDA(EPOS):

    def __init__(self, N=35, Nexpp_coarse=50, Nexpp_fine=200, counter_level=[3,9,14,20],
                    results_path='/home/bermuda/Results/tiamo3.sync',
                    wvf_folder='../UserData', \
                    wvf_db_file='../UserData/waveform_db.json', \
                    cache_path='./data/', log_file='./data/log.txt', do_plot=True, do_live=False, dark=False):
        super().__init__(cache_path=cache_path, log_file=log_file, do_plot=do_plot, do_live=do_live, wvf_folder=wvf_folder, wvf_db_file=wvf_db_file)

        self.N = N
        self.Nexpp_coarse = Nexpp_coarse
        self.Nexpp_fine = Nexpp_fine
        self.counter_level = counter_level
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

    def create_eios_script(self, seq = ['PDQ_init','COOL_d', 'DET_bdx'], verbose=True, s_hlp=False):
        import pandas as pd
        pb_scrpt_prts_pd=pd.read_csv("../UserData/script_lines_2019_09_05.csv", index_col=0)

        txt = pb_scrpt_prts_pd.at['INCLDS', 'Script line']
        txt += '\n'+pb_scrpt_prts_pd.at['HDR', 'Script line']
        for pls in seq:
            txt += '\n\n //---'+pls+'---\n'+pb_scrpt_prts_pd.at[pls, 'Script line']+'\n //---------------'
        txt += '\n'+pb_scrpt_prts_pd.at['FTR', 'Script line']
        script_file='../UserData/Scripts/Basic_Experiments/remote.pre'

        with open(script_file, 'w') as outfile:
            outfile.write(txt)

        if verbose:
            print('Scirpt:',script_file)
            print('-------------------------BEGIN-------------------------')
            txt=''
            with open(script_file, 'r') as fin:
                txt += fin.read()
            print(txt)
            print('-------------------------END-------------------------')
        return 'remote'

    def create_scan_para(self, par_type='fr', name='1_rsb_1', scl=1., start=None, stop=None, npts=None, nexp=None, fit_func=None, invert=None):
        if par_type=='fr':
            par_name=par_type+'_'+name
            cnt=float(self.get(par_type+'_'+name))
            dur=float(self.get('t_'+name))
            if start==None:
                start=cnt-scl/dur
            if stop==None:
                stop=cnt+scl/dur
            if npts==None:
                npts=int(scl*11)
            if fit_func==None:
                fit_func=self.fit_single_sinc
            if invert==None:
                invert=True

        if par_type=='t':
            par_name=par_type+'_'+name
            dur=float(self.get('t_'+name))
            if start==None:
                start=0
            if stop==None:
                stop=scl*2*dur
            if npts==None:
                npts=int(scl*12)
            if fit_func==None:
                fit_func=self.fit_time
            if invert==None:
                invert=False

        if par_type=='phs':
            par_name=par_type+'_'+name
            dur=float(self.get('phs_'+name))
            if start==None:
                start=-scl*np.pi
            if stop==None:
                stop=scl*np.pi
            if npts==None:
                npts=int(scl*12)
            if fit_func==None:
                fit_func=self.fit_phase
            if invert==None:
                invert=False

        if par_type==None:
            par_name=name
            if start==None:
                start=0
            if stop==None:
                stop=scl*1
            if npts==None:
                npts=int(scl*5)
            if fit_func==None:
                fit_func=None
            if invert==None:
                invert=False

        if nexp==None:
            nexp=200

        return par_name, start, stop, npts, nexp, fit_func, invert

    def set_ips(self, ips=None):
        if ips != None:
            for ip in ips:
                self.set(ip[0],str(ip[1]))

    def run_seq(self, seq=['PDQ_init', 'COOL_d', 'DTCT_bdx'],
                par_type=None, par_name='dummy', scl=1.0, ips=None, start=None, stop=None, npts=None, nexp=None,
                fit_func=None, invert=None, fit_result=True, set_op_para=False, check_script=False):
        '''
        Define your experimental PAULBOX sequence and
        run it with default (or your choice of) parameters
        '''
        script_name=self.create_eios_script(seq=seq, verbose=check_script);
        par_name, start, stop, npts, nexp, fit_func, invert=self.create_scan_para(par_type=par_type,name=par_name, scl=scl, start=start, stop=stop, npts=npts, nexp=nexp, fit_func=fit_func, invert=invert)

        if check_script==True:
            print('Scan_para: %s \n Start: %.5f, \n Stop:: %.5f\n N_pts: %i \n N_exp: %i \n Fit_func: %s \n Fit_invert: %s'%(par_name, start, stop, npts, nexp, fit_func, invert))
           #print('s')
        if check_script==False:
            self.set_ips(ips=ips)
            if self.get('spin_up_mw')=='1' and invert==True:
                invert=False
            else:
                if self.get('spin_up_mw')=='1' and invert==False:
                    invert=True
            descr='Experimental sequence:'
            for pulse in seq:
                if pulse=='header' or pulse=='footer':
                    descr+='\n'
                else:
                    descr+='\n *'+pulse
            if set_op_para==True:
                name_var = par_name
            else:
                name_var=''
            if fit_func==None or fit_result==False:
                name, data = self.run(script_name=script_name, par_name=par_name, start=start, stop=stop, numofpoints=npts, expperpoint=nexp)
                self.plt_raw_dt(data=data, name=name, xlabel=par_name+' (a.u.)', ylabel='Cts.', text=descr)
                return name, data
            else:
                return self.single_fit_run(script_name=script_name, par_name=par_name, start=start, stop=stop, numofpoints=npts, expperpoint=nexp, invert=invert, xlabel=par_name+' (a.u.)', ylabel='Cts.', name_var=name_var, add_contrast=True, give_all_para=True, fit_func=fit_func, text=descr)
        else:
            if par_type==None:
                return 'Not run...', [[0.0],[0.0],[0.0]]
            else:
                return 0.0, 0.0, 0.0, [[0.0],[0.0],[0.0]], [[0.0],[0.0],[0.0]], [0.0]

    def check_ion(self, verbose=True, numofpoints=0, expperpoint=250, counter=2):
        _,data=self.run('BDX2', numofpoints=numofpoints, expperpoint=expperpoint, verbose=False, counter=counter)
        print(data)
        cnt = data[1][-1]
        if verbose:
            print('BDx cts:', cnt)
        return cnt

    def count_ions(self, verbose=False, give_all_param=False):
        cnts = self.check_ion(verbose=False,numofpoints=1)
        N=-1
        for i,l in enumerate(self.counter_level):
            if cnts<l:
                N=i
                break
        if verbose:
            print('Counted %i ion(s) (BDX cnt %.2f)!'%(N, cnts))
        if give_all_param:
            return N,cnts
        else:
            return N

    def switch_to_pmt(self, state, ch_pmt=0):
        self.ttl_ME(ch_pmt, not state)

    def open_photo_laser(self, state, ch_load=7):
        self.ttl_ME(ch_load,state)

    def heat_oven(self, state, ch_oven_0=4, ch_oven_1=5):
        self.ttl_ME(ch_oven_1,0)
        self.ttl_ME(ch_oven_0,state)
        time.sleep(1.)

    def get_all_ips(self, loc=None):
        import pandas as pd
        import PyModules.analyse_eios.eios_data as eios_file

        name, data = self.run('BDX', numofpoints=0, expperpoint=1, verbose=False)
        data, xml = eios_file.read(name, sort=True, skip_first=False)
        xml_dict = eios_file.read_xml(xml)

        ip_names=xml_dict['ionproperties']
        ip_names=list(ip_names.keys())
        #print(ip_names)
        #print(len(ip_names))
        #ip_names = self.list_ip()
        #print(loc in ip_names)
        #print(len(ip_names))
        #print(ip_names)
        ips=[]
        vls=[]
        for ip_name in ip_names:
            try:
        #        val, set_date = euro.get_parameter(ip_name)
        #        ips.append([ip_name, val, set_date])
                val = self.get(ip_name)
                if val != '':
                    ips.append(ip_name)
                    vls.append(val)
            except:
                0 #print(ip_name+' does not exist!')
        #print(ips, vls)
        ips_pd = pd.DataFrame(vls, index=ips, columns=['Setting'])
        if loc != None:
            print(ips_pd.loc[loc: loc+'zzz', ['Setting']])
        return ips_pd

    def get_rf_pwr(self, NoAvg=10):
        if NoAvg == 1:
            pwr_rf=self.read_adc_me(31)
            self.set('rfpwr',str(pwr_rf))
            return pwr_rf
        else:
            pwr_rf, err=self.read_avg_adc_me(ch=31, no_avg=NoAvg, verbose=False)
            self.set('rfpwr',str(pwr_rf))
            return pwr_rf, err

    def cal_modefr(self, script, mode_name, amp, dur, el_num = 26, expperpoint=250, numofpoints=50, rng= 2.25 ,**kwargs):
        self.set('t_tickle',str(dur))
        self.set('u_tickle',str(amp))
        fr = float(self.get('fr_'+mode_name))
        self.set('shim_tickle', el_num)
        start = fr-rng/dur
        stop = fr+rng/dur
        return self.single_fit_frequency(script, 'fr_'+mode_name, invert=True, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, live_plots_flag=False, give_all_para=True, **kwargs)

    def tickle_ramsey(self, script, mode_name, amp, dur, twait, el_num, expperpoint=250, numofpoints=11, rnd_sampled=False, **kwargs):
        fr=float(self.get('fr_'+mode_name))
        self.set('fr_tickle',str(fr))
        self.set('tickle_div',str(1))
        self.set('t_tickle',str(dur/2))
        self.set('t_tickle_'+mode_name,str(dur));
        self.set('u_tickle',str(amp))
        self.set('u_tickle_'+mode_name,str(amp));
        self.set('t_tickle_wait',str(twait));
        start=-.65
        stop=.65
        #el_num=self.get('shim_tickle_'+mode_name);
        self.set('shim_tickle', el_num);
        popt, perr = [0,0,0,0,0], [0,0,0,0,0]
        if self.count_ions()>0:
            val, err, R2, popt, perr, chi2 = self.single_fit_phase_pi(script, 'dummy', invert=True, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, live_plots_flag=False,  give_all_para=True, rnd_sampled=rnd_sampled, **kwargs)
        return popt, perr, R2


    def get_mode_coh(self, mode_name, amp, dur, Ramsey_durs, el_num, counter, expperpoint=250, numofpoints=11, rnd_sampled=False, start=[1e4, 2]):
        cal_begin=self.print_header('Mode #'+mode_name+' coherence')
        res=np.array([])
        self.cal_modefr('PDQ_Tickle_BDX2', mode_name, amp, dur, el_num, \
                        expperpoint=expperpoint, numofpoints=14, rng=2.5)
        n=0
        for i in Ramsey_durs:
            if self.count_ions()>0:
                print('------------------------------')
                print('Ramsey duration (µs): ',i)
                print('------------------------------')
                self.cal_modefr('PDQ_Tickle_BDX2', mode_name, amp, dur, el_num, \
                                expperpoint=expperpoint, numofpoints=14, rng=3)
                popt, perr, R2 = self.tickle_ramsey('PDQ_Tickle_phasescan', mode_name, \
                                                    amp, dur, i, expperpoint=expperpoint, \
                                                    numofpoints=numofpoints, rnd_sampled=rnd_sampled)
                if R2>.005:
                    n=n+1
                    res=np.append(res,(float(i), np.abs(popt[0]),perr[0]))
                    rres=res.reshape((n,3))
                    data=[rres[:,0], rres[:,1]/np.max(rres[:,1]), 0.5*rres[:,2]/np.max(rres[:,1])]
                if n>1:
                    self.plot_data(data)
                    plt.xlabel('Ramsey dur. (µs)')
                    plt.ylabel('Contrast (a.u.)')
                    plt.show()

        self.print_footer(cal_begin)

        self.print_header('Analyse mode #'+mode_name+' coherence')
        try:
            lbl = ['Ramsey dur. (µs)','Contrast (a.u.)','Mode #'+mode_name, 'Model: Decay w. exp']
            popt, perr, R2, chi2 = self.fit_data(func_decay_exponential, data, start, plt_labels=lbl);
        except:
            popt = np.array(start)
            perr = 0.0*popt
            R2, chi2 = 0., 0.
        self.print_footer(cal_begin)

        return data, popt, perr, R2, chi2
