#!/usr/bin/env python3

import numpy as np
import scipy as sci
from scipy import stats, signal
from scipy.stats import norm

import os.path
import time
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import allantools

from ipywidgets import Output
from IPython.display import clear_output, display

import random
import string

from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save
from PyModules.analyse_eios.eios_sb import LDparameter, fit_flop_sb, fit_flop_sb_fock, fit_flop_carr_bsb, fit_flop_carr_bsb_fock, fit_flop_sb_fock_rob, fit_dist_fock, fit_flop_carrier, plot_fock_fit, plot_flop_fit, open_file

from PyModules.analyse_eios.eios_analyse import unpack_sorted, significant_digit, plot_fit
from PyModules.analyse_eios.eios_analyse import fit_direct, fit_linear, fit_parameter, fit_func
from PyModules.analyse_eios.eios_analyse import fit_multi_freq, fit_time, fit_phase
from PyModules.analyse_eios.eios_analyse import gauss_sum, lorentz_sum, abs_sum, sinc_sum, parabola_sum, sinc_abs_sum, sincSquare, gauss, lorentz, parabola

from PyModules.utilities import hash

from PyModules.utilities import do_async, integer_hill_climb, wait
from PyModules.wavemeter.pid_wrapper import pid_container
from PyModules.wavemeter.lock_client import web_lock_client

#from PyModules.machine_learning.run_ml import write_waveform
#from PyModules.pdq.waveform import smooth_step_eu, b_ex_1, b_ex_2, d_b_ex_2, linear_ramp

def RandomId(stringLength=6):
    letters = string.ascii_letters + string.digits
    return ''.join([random.choice(letters) for i in range(stringLength)])

# Experiment Python Operating System
class EPOS_V:

    def __init__(self, cache_path='./data/', log_file='./data/log.txt', log_file_org='./data/log.txt', \
                        gof_thres = .7, do_plot=True, do_live=False, \
                        wvf_folder='../UserData', wvf_db_file='../UserData/waveform_db.json'):

        self.last_script_name = ''
        self.session_id = None

        self.cache_path = cache_path
        self.log_file = log_file
        self.log_file_org = log_file_org
        self.gof_thres = gof_thres

        self.do_plot = do_plot

        self.wvf_folder = wvf_folder
        self.wvf_db_file = wvf_db_file

        self.fit_single_gauss = lambda x,y,y_err: fit_multi_freq(gauss,x,y,y_err, N=1)
        self.fit_single_gauss.__name__ = "fit_single_gauss"
        self.fit_three_gauss = lambda x,y,y_err: fit_multi_freq(gauss_sum,x,y,y_err,N=3)
        self.fit_three_gauss.__name__ = "fit_three_gauss"
        self.fit_multi_gauss = lambda x,y,y_err: fit_multi_freq(gauss_sum,x,y,y_err)
        self.fit_multi_gauss.__name__ = "fit_multi_gauss"

        self.fit_single_lorentz = lambda x,y,y_err: fit_multi_freq(lorentz,x,y,y_err,N=1)
        self.fit_single_lorentz.__name__ = "fit_single_lorentz"
        self.fit_multi_lorentz = lambda x,y,y_err: fit_multi_freq(lorentz_sum,x,y,y_err)
        self.fit_multi_lorentz.__name__ = "fit_multi_lorentz"

        self.fit_single_sinc = lambda x,y,y_err: fit_multi_freq(sincSquare,x,y,y_err,N=1)
        self.fit_single_sinc.__name__ = "fit_single_sinc"
        self.fit_multi_sinc = lambda x,y,y_err: fit_multi_freq(sinc_sum,x,y,y_err)
        self.fit_multi_sinc.__name__ = "fit_multi_sinc"

        self.fit_single_parabola = lambda x,y,y_err: fit_multi_freq(parabola,x,y,y_err,N=1)
        self.fit_single_parabola.__name__ = "fit_single_parabola"
        self.fit_single_abs = lambda x,y,y_err: fit_multi_freq(abs_sum,x,y,y_err,N=1)
        self.fit_single_abs.__name__ = "fit_single_abs"
        self.fit_single_sinc_abs = lambda x,y,y_err: fit_multi_freq(sinc_abs_sum,x,y,y_err,N=1)
        self.fit_single_sinc_abs.__name__ = "fit_single_sinc_abs"

        self.fit_time = lambda x,y,y_err: fit_time(x,y,y_err)
        self.fit_time.__name__ = "fit_time"

        self.fit_phase = lambda x,y,y_err: fit_phase(x,y,y_err,f=1.,f_min=1.,f_max=np.pi)
        self.fit_phase.__name__ = "fit_phase"

        self.fit_phase_pi = lambda x,y,y_err: fit_phase(x,y,y_err,f=np.pi,f_min=0.9*np.pi,f_max=1.1*np.pi)
        self.fit_phase_pi.__name__ = "fit_phase_pi"

        self.fit_linear = lambda x,y,y_err: fit_linear(x,y,y_err)
        self.fit_linear.__name__ = "fit_linear"

    def get_wvm_status(self):
        wlc = web_lock_client(host='10.5.78.145', port=8000)
        #print(wlc.get_status())
        stat, ch_list=wlc.get_list()
        for i in [0,1,2,3,4,5]:
            self.show_log_data(ch_list[i], wlc)

    def _str2ts(self, t_str, str_format='%Y-%m-%d %H:%M:%S'):
        return datetime.datetime.strptime(t_str, str_format).timestamp()

    def _ts2str(self, t_ts, str_format='%Y-%m-%d %H:%M:%S'):
        if isinstance(t_ts, (int, float)):
            t_ts = time.localtime(t_ts)
        return time.strftime(str_format,t_ts)

    def _log(self,msg):
        if self.log_file is not None:
            ts = self._ts2str(time.localtime())
            with open(self.log_file, 'a+') as f:
                f.write('%s %s\n'%(ts,msg))

    def copy_latest_log_file(self):
        import sys
        import shutil

        dest = self.log_file
        src = self.log_file_org
        if dest != src:
            print('-------------------------------------')
            print('Get latest log file... please wait...')
            if (not os.path.exists(dest)) or (os.stat(src).st_mtime - os.stat(dest).st_mtime > 1) :
                shutil.copy2 (src, dest)
                print('File updated-'+self._ts2str(os.stat(src).st_mtime))
            else:
                print('File not updated - Version: '+self._ts2str(os.stat(src).st_mtime))
            print('-------------------------------------')

    def get_log_index(self, index_start, index_stop):
        data = []
        if self.log_file is not None:
            with open(self.log_file) as log_file:
                for i,line in enumerate(reversed(log_file.readlines())):
                    if i>index_stop:
                        if i<index_start:
                            data.append(line.rstrip().split())
                        else:
                            break
        data.reverse()
        return data

    def get_log(self, key, name, s_start, s_stop):
        data = []
        if self.log_file is not None:
            t_start = self._str2ts(s_start)
            t_stop = self._str2ts(s_stop)
            for i,line in enumerate(reversed(open(self.log_file).readlines())):
                line_strip = line.rstrip()
                line_split = line_strip.split()
                #print(i, '"%s"'%line_strip)
                if len(line_split)>3 and line_split[2] == key:
                    #print(i, '%s = %s (%s %s)'%(line_split[3],line_split[4],line_split[0],line_split[1]))
                    if line_split[3] == name:
                        ret_date = line_split[0]
                        ret_time = line_split[1]
                        ret_tstamp = self._str2ts('%s %s'%(ret_date,ret_time))
                        #print(ret_date,ret_time)
                        ret_value = line_split[4:]
                        #print(ret_tstamp)
                        if (ret_tstamp>t_start):
                            #print(ret_tstamp)
                            if (ret_tstamp<t_stop):
                                data.append([ret_tstamp, *ret_value])
                        else:
                            break
            data.reverse()
        return data

    def find_log(self, key, name, value=None, timeout=None):
        if self.log_file is not None:
            for i,line in enumerate(reversed(open(self.log_file).readlines())):
                line_strip = line.rstrip()
                line_split = line_strip.split()
                #print(i, '"%s"'%line_split)
                if line_split[2] == key:
                    #print(i, '%s = %s (%s %s)'%(line_split[3],line_split[4],line_split[0],line_split[1]))
                    if line_split[3] == name:
                        ret_date = line_split[0]
                        ret_time = line_split[1]
                        if (value is None) or (line_split[4] == value):
                            return True, i, ret_date, ret_time
                if timeout is not None and i>timeout:
                    break
        return False, i, '', ''

    def print_header(self, txt):
        cal_begin=datetime.datetime.now()
        print('----------------------------------')
        print(txt)
        print(str(cal_begin))
        print('Please wait...')
        print('---------------------------------- \n')
        return cal_begin

    def print_footer(self, cal_begin):
        cal_end = datetime.datetime.now()
        cal_dur = cal_end-cal_begin
        print('----------------------------------')
        print('Done. Seq. took (hours:min:sec):')
        print(str(cal_dur))
        print('---------------------------------- \n')
        return cal_dur

    def show_ips(self, name, ips):
        _, root = read(name)
        ip = read_xml(root)['ionproperties']
        text_ips='\nIon props:\n'
        ip_l = []
        for i in ips:
            text_ips+= str(i) +' = '+str(ip[i])+'\n'
            ip_l.append([str(i),ip[i]])
        return ip_l, text_ips

    def show_script(self, name):
        _, root = read(name)
        script = read_xml(root)['sourcecode']
        scan_para = read_xml(root)['parameter']['name']
        del_until=script.find('pdq_init(')
        del_from=script.find('read();')
        script=script[del_until:del_from]
        script=script.replace('\n\n\n\n','\n').replace('\n\n\n','\n').replace('\n\n','\n')
        return script, scan_para

    def show_data(self, run_times, ips=[''], verbose=True):
        """
        show_data(run_times, ips=[''], verbose=True)

        Displays data and script details.

        Returns file names, data, scripts,
        and requested ion properties for further use

        Parameters
        ----------
        run_times : list of strings
            with format ['Hour_Min_Sec_Day_Month_Year.dat'],
            e.g., ['10_45_49_13_03_2020.dat']

        ips : list of strings
            Choose which ion property values to return

        Returns
        -------
        name_l, data_l, script_l, ips_l
            Lists of file names, data, scripts and req. ion properties.
        """

        path_data_local = self.results_path
        #print(path_data_local)
        times=[]
        for i in range(len(run_times)):
            times.append(run_times[i])
        name_l=find_files(times, eios_data_path = path_data_local)
        print('Files: ', name_l)
        data_l=[]
        script_l=[]
        ips_l=[]
        for l in range(len(name_l)):
            name=name_l[l]
            data=read(name)
            data_l.append(data)
            script, scan_para = self.show_script(name)
            script_l.append(script)
            ip_l, text_ips=self.show_ips(name, ips)
            ips_l.append(ip_l)
            raw_data_avg=[]
            for k in range(len(data[0])):
                raw_data_avg.append([data[0][k]['x'],data[0][k]['y'],data[0][k]['error']])
            if verbose==True:
                if len(raw_data_avg)>0:
                    fig, ax = plt.subplots()
                    ColorList=['Navy','Red','Orange','Grey','Silver', 'Black']
                    if name:
                        fig.canvas.set_window_title(name)
                    for i, data in enumerate(raw_data_avg):
                        x, y, y_err = data
                        plt.errorbar(x, y, yerr = y_err, linestyle = "None", marker = "o", label='Det.# %i'%i, color=ColorList[i], markersize=7.5, lw=1., capsize=.0)
                    plt.legend(loc='upper right')
                plt.title('.'+name[(len(path_data_local)+5):])
                plt.ylabel('Cts.')
                plt.xlabel(scan_para+' (a.u.)')
                plt.text(1.1, .5, 'Experimental sequence:\n\n'+script_l[l]+'\n'+text_ips, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
                plt.show()
        return name_l, data_l, script_l, ips_l

    def session_get(self, session_id):
        data = []
        stat_stop = False
        stat_start, idx_start, ret_date, ret_time = self.find_log(key='session', name=session_id, value='start')
        if stat_start:
            stat_stop, idx_stop, ret_date, ret_time = self.find_log(key='session', name=session_id, value='stop', timeout=idx_start)
            if stat_stop:
                data = self.get_log_index(idx_start, idx_stop)
        return stat_start, stat_stop, data

    def get_session_list(self, file_name, date_search):
        line_number = 0
        list_of_results = []
        with open(file_name, 'r') as read_obj:
            for line in read_obj:
                line_number += 1
                if date_search in line and 'session' in line and 'start' in line:
                    list_of_results.append((line.rstrip()[28:47]))
        session_l = np.sort([*set(list_of_results), ])
        print('Found the follwoing sessions:')
        for session_id in session_l:
            logs, annotations, data_key = self.session_find(session_id)
            txt = str(session_id)
            if annotations != []:
                txt +=' - '+str(annotations[0][2])
            print(txt)
        return session_l

    def get_run_list(self, file_name, date_search, script=''):
        line_number = 0
        list_of_results = []
        with open(file_name, 'r') as read_obj:
            for line in read_obj:
                line_number += 1
                if date_search in line:
                    if 'run' in line and script in line and not 'BDX' in line:
                        list_of_results.append((line))
        run_l = np.sort([*set(list_of_results), ])
        print('Found the follwoing runs:')
        name_l = []
        for run in run_l:
            head, sep, tail = run.partition('.dat')
            txt = head+sep
            #if annotations != []:
            #    txt +=' - '+str(annotations[0][2])
            name_l.append(head[-19:]+sep)
            print(txt)
        return name_l

    def session_parse(self, data, key):
        logs = []
        annotations = []
        data_key = []
        j = 0
        for i,line in enumerate(data):
            if len(line)>4 and line[4]=='annotate':
                annotations.append([*line[:2], ' '.join(line[5:])])
                logs.append(data[j:i])
                j=i+1
            elif key is None or (len(line)>2 and line[2]==key):
                data_key.append(line)
                logs.append(data[j:i+1])
        return logs, annotations, data_key

    def session_find(self, session_id, key=None):
        stat_start, stat_stop, data = self.session_get(session_id)
        logs, annotations, data_key = self.session_parse(data, key)
        return logs, annotations, data_key

    def session_replay(self, session_id, ips_l=[''], show_details=False, key="fit", do_plot=True):
        txt = 'Loading lab notes from: '+session_id;
        logs, annotations, data_key = self.session_find(session_id, key)
        if annotations != []:
            txt += '\n'+str(annotations[0][2]);
        cal_begin = self.print_header(txt)
        #print(annotations[3])
        ret_list = []
        data_list=[]

        for d_key in data_key:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            script_name = d_key[3]
            #########################
            if script_name=='heating_rates_iteration':
                sids = []
                return d_key[4]
            #########################
            filename = d_key[4].replace('/home/qsim/Results/tiamo4.sync', self.results_path).replace('/home/bermuda/Results_H/tiamo3.sync', self.results_path).replace('/home/bermuda/Results/tiamo3.sync', self.results_path)
            name_var = d_key[5]
            func_str = d_key[6]
            ###########################
            if script_name=='heating_rates':
                data, _ = read(filename, sort=True, skip_first=False)
                return data
            ##################################
            func = getattr(self, func_str)
            if script_name=='single_fit_sb':
                fullpath = filename[1:-1].split(',')
                fock = bool(int(d_key[7]))
                mode_freq = float(d_key[8])
                mode_angle = float(d_key[9])
                Rabi_init = float(d_key[10])
                redflop, blueflop, lbl = open_file(fullpath, self.cache_path)
                ret = func(redflop, blueflop, lbl, mode_freq=mode_freq, mode_angle=mode_angle, Rabi_init=Rabi_init)
                ret_list.append(ret)
            else:
                invert = bool(int(d_key[7]))
                counter = int(d_key[8])
                title = ' '.join(d_key[9:])

                data, _ = read(filename, sort=True, skip_first=False)
                if show_details:
                    print(str('Experimental seq.:\n\t'+self.show_script(filename)[0]))
                    print(self.show_ips(filename, ips_l)[1])
                for i,d in enumerate(data):
                    if i == counter:
                        self.do_plot=do_plot
                        ret = self.single_fit_func([d['x'], d['y'], d['error']], func, invert=invert, \
                                                       give_all_para=True, name_var=name_var, \
                                                       xlabel=name_var+'(a.u.)', ylabel='Fluo. cts.', add_contrast=False, \
                                                       text=None, title=title, set_fit=False)
                        ret_list.append(ret)
                        data_list.append(data)

        self.print_footer(cal_begin)
        return ret_list, data_list

    def read_ip_hstry(self, ip, rng, verbose=True):
        data = self.get_log('set', ip, rng[0], rng[1])
        for i,d in enumerate(data):
            data[i] = [d[0], float(d[1]), float(d[2][1:-1])]
            data[i] = [*data[i], data[i][1]]
        data = np.array(data)

        if verbose == True and len(data)==0:
            print('No data available')

        if verbose == True and len(data)>1:
            x = (data[:,0]-data[0, 0])
            y = data[:,3]
            sampl_rate = ((data[-1,0]-data[0,0])/len(data[:,3]))**(-1)
            f, Pxx_den = signal.periodogram(y, sampl_rate)
            if len(data)>2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                txt = self._ts2str(data[0, 0])
                txt += ' [Total dur. (min): '+str(round((data[-1,0]-data[0, 0])/60,1))+' / Sample size: '+str(len(y))+']'
                txt += '\nMean: '+self.prnt_rslts(ip, np.mean(y), np.std(y), verbose=False)
                txt += '\nDrift: '+ self.prnt_rslts('d/dt', slope*60*1e3, std_err*60*1e3, verbose=False)+'$\,\cdot\,10^{-3}$ min$^{-1}$'
                plt.axes([.0, .12, .72, .76])
                plt.plot(x, y,
                         ls='--', marker = 'o', markersize=5., color='navy')
                plt.xlabel('Duration (s)')
                plt.ylabel(ip+' (a.u.)')
                plt.title(txt)
                sns.regplot(x, y, color='red',
 line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
            if len(data)>10:
                mu, std = norm.fit(y)
                xmin=mu-2.5*std
                xmax=mu+2.5*std
                n=max(significant_digit(std),0)
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.axes([.77, .12, .25, .30])
                plt.hist(y, bins = int(2*np.log(len(y))), density=True, color='navy', edgecolor='white')
                plt.plot(x, p, color='red', linewidth=2, alpha=0.75)
                plt.xlim((xmin, xmax))
                plt.xlabel(ip+' (a.u.)')
                plt.ylabel('Prop.')
                plt.xticks([round(mu-1.5*std,n), round(mu+1.5*std,n)])
                plt.yticks([])
                plt.axes([.77, .58, .25, .30])
                plt.semilogy(f, Pxx_den, color='navy')
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('PSD (a.u.)')
                plt.yticks([])
                #plt.grid(False)
                plt.ylim(1e-3*np.mean(Pxx_den), 2*np.max(Pxx_den))

            plt.savefig('last_longterm_'+ip.replace('.','_'), bbox_inches='tight')
            plt.show()
        return data

    def eval_oadev(self, freq_data, sampl_rate, lbl='Freq. data', units='Hz', show_ref='None', scl_ref=1, verbose=True, rel=1):
        if rel==1:
            freq_data=(freq_data-np.mean(freq_data))/np.mean(freq_data)
        a = allantools.Dataset(data=freq_data, rate=sampl_rate, data_type='freq', taus='all')
        res=a.compute("oadev")

        if verbose==True:

            if show_ref!='None':
                res_ref_l=np.array([])
                n=0
                for i in range(200):
                    n=n+1
                    if show_ref=='Pink':
                        yref=scl_ref*allantools.noise.pink(len(freq_data))
                        plt_color=show_ref
                    if show_ref=='Brownian':
                        yref=scl_ref*allantools.noise.brown(len(freq_data), fs=sampl_rate)
                        plt_color='brown'
                    if show_ref=='White':
                        yref=scl_ref*allantools.noise.white(len(freq_data), fs=sampl_rate)
                        plt_color='grey'
                    if show_ref=='Violet':
                        yref=scl_ref*allantools.noise.violet(len(freq_data))
                        plt_color=show_ref

                    aref = allantools.Dataset(data=yref, rate=sampl_rate,
                                              data_type='freq', taus='all')
                    res_ref=aref.compute("oadev")
                    res_ref_l=np.append(res_ref_l,res_ref['stat'])
                    coll_res_ref=res_ref_l.reshape(n,len(res_ref['stat']))
                    coll_res_ref_scld=coll_res_ref
                    mean_noise=np.mean(coll_res_ref_scld, axis=0)
                    std_noise=np.std(coll_res_ref_scld, axis=0)
            plt.axes([.0, .0, .98, .85])
            plt.errorbar(res['taus'], res['stat'], yerr=res['stat_err'], label=lbl, marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True,which="both",ls="-")
            plt.grid(True, which='minor', alpha=0.5)
            if show_ref!='None':
                plt.loglog(res['taus'], mean_noise, color=plt_color, label=show_ref+' noise')
                plt.fill_between(res['taus'], mean_noise+std_noise/2, mean_noise-std_noise/2, color=plt_color, alpha=.5)
            plt.xlabel('Avg. duration (s)')
            if rel==1:
                plt.ylabel('Overlapping Allan dev.\n$\Delta$'+lbl+'/'+lbl)
            else:
                plt.ylabel('Overlapping Allan dev. ('+units+')')
                plt.legend()
            plt.savefig('last_allan_'+lbl.replace('.','_'), bbox_inches='tight')
            plt.show()
            #b.show()
        return a

    def plot_data(self, data, name='', lbls=['','']):
        data = np.array(data)
        if len(data.shape)<3:
            data = np.array([data])
        return super().plot_data(name, data, lbls=lbls)

    def errplt(self, res, lbls=['Scan para (a.u.)','Results (a.u.)'], comp=[None, None]):
        npres=np.array(res)
        plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.0125)
        ax=plt.subplot(221)
        plt.errorbar(npres[:,0], npres[:,1], yerr=npres[:,2], label='', marker = 'o', color='navy', markersize=7.5, lw=1., capsize=.0, ls='', fmt='')
        if comp!= [None, None]:
            plt.plot(comp[0], comp[1])
        plt.grid(True)
        plt.xlabel(lbls[0])
        plt.ylabel(lbls[1])
        plt.show()

    def single_fit_carrier_data(self, flop, lbl, mode_freq, mode_angle, Rabi_init, dec_init=0.001, lim_init=0.85, nth=0.1, ncoh=0.001, nsq=0.001, fix=[0,0,0,0,1,1], nmax=20, ntrot=1):
        LD = LDparameter(mode_freq,mode_angle)
        initparams = [Rabi_init,dec_init,lim_init, nth, ncoh, nsq]
        red_chi, fmin, param, m, flop_func_list, value, error, \
            fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = \
                fit_flop_carrier(flop, LD, nmax, initparams, fix, ntrot)

        [fit_rabi, fit_dec, fit_lim, fit_nth, fit_ncoh, fit_nsq] = value
        [fit_rabi_err,fit_dec_err,fit_lim_err,fit_nth_err,fit_ncoh_err,fit_nsq_err] = error
        fit_valid = fmin['is_valid']
        if self.do_plot:
            if fit_valid:
                fit_status = '$n_{th}$\t= %.4f +- %.4f\n$n_{coh}$\t= %.4f +- %.4f\n$n_{sq}$\t= %.4f +- %.4f' % (fit_nth,fit_nth_err,fit_ncoh,fit_ncoh_err,fit_nsq,fit_nsq_err)
            else:
                fit_status = 'fit failed'
            plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [flop], lbl, fit_status, figsize=(10,4));
            plt.show()

        for (key, val),( _, err),f in zip(m.values.items(), m.errors.items(), fix):
            n = max(significant_digit(err),0)
            s_v = '%%%i.%if'%(6+n,n)
            s_e = '%%%i.%if'%(1+n,n)
            txt_par = '%s\t= %s +- %s'%(key,s_v%val,s_e%err)
            if f>0:
                txt_par += '*'
            print(txt_par)

        return red_chi, fit_valid, value, error

    def single_fit_sb_data(self, redflop, blueflop, lbl, mode_freq, mode_angle, Rabi_init, dec_init=0.001, limb_init=0.5, limr_init=0.5, nth=0.1, ncoh=1e-9, nsq=1e-9, fix=[0,0,0,0,0,1,1], nmax=20, ntrot=1):
        LD = LDparameter(mode_freq,mode_angle)
        initparams = [Rabi_init,dec_init,limb_init,limr_init, nth, ncoh, nsq]

        red_chi, fmin, param, m, flop_func_list, value, error,\
            fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = \
                fit_flop_sb(redflop, blueflop, LD, nmax, initparams, fix, ntrot)

        [fit_rabi, fit_dec, fit_limb, fit_limr, fit_nth, fit_ncoh, fit_nsq] = value
        [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err,fit_nth_err,fit_ncoh_err,fit_nsq_err] = error
        fit_valid = fmin['is_valid']

        if fit_valid:
            #fit_status = '$n_{th}$\t= %.4f +- %.4f\n$n_{coh}$\t= %.4f +- %.4f\n$n_{sq}$\t= %.4f +- %.4f' % (fit_nth,fit_nth_err,fit_ncoh,fit_ncoh_err,fit_nsq,fit_nsq_err)
            fit_status = '$red. \chi^2$= %.3f\n $\Omega_{0}$= %.3f +- %.3f\n $\Gamma_{dec}$= %.3f +- %.3f\n $n_{th}$= %.3f +- %.3f\n$n_{coh}$= %.3f +- %.3f\n$n_{sq}$= %.3f +- %.3f' % (red_chi, fit_rabi, fit_rabi_err, fit_dec, fit_dec_err, fit_nth, fit_nth_err, fit_ncoh, fit_ncoh_err, fit_nsq, fit_nsq_err)
        else:
            fit_status = 'fit failed'
        if self.do_plot:
            plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [redflop, blueflop], lbl, fit_status);
            plt.show()

        for (key, val),( _, err),f in zip(m.values.items(), m.errors.items(), fix):
            n = max(significant_digit(err),0)
            s_v = '%%%i.%if'%(6+n,n)
            s_e = '%%%i.%if'%(1+n,n)
            txt_par = '%s\t= %s +- %s'%(key,s_v%val,s_e%err)
            if f>0:
                txt_par += '*'
            #print(txt_par)

        return red_chi, fit_valid, value, error


    # from rob, for carrier sideband fit
    def single_fit_sb_carr_data(self, carrflop, blueflop, lbl, bluescale, mode_freq, mode_angle, Rabi_init, decc_init=0.001, decb_init=0.001, limc_init=0.5, limb_init=0.5, nth=0.1, ncoh=1e-9, nsq=1e-9, fix=[0,0,0,0,0,1,1], nmax=20, ntrot=1):
        LD = LDparameter(mode_freq,mode_angle)
        initparams = [Rabi_init,decc_init, decb_init, limc_init,limb_init, nth, ncoh, nsq]

        red_chi, fmin, param, m, flop_func_list, value, error,\
            fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = \
                fit_flop_carr_bsb(carrflop, blueflop, LD, nmax, initparams, fix, ntrot, bluescale)

        [fit_rabi, fit_decc, fit_decb, fit_limc, fit_limb, fit_nth, fit_ncoh, fit_nsq] = value
        [fit_rabi_err,fit_decc_err,fit_decb_err,fit_limc_err,fit_limb_err,fit_nth_err,fit_ncoh_err,fit_nsq_err] = error
        fit_valid = fmin['is_valid']

        if fit_valid:
            fit_status = '$red. \chi^2$= %.3f\n $\Omega_{0}$= %.3f +- %.3f\n $\Gamma_{dec,C}$= %.3f +- %.3f\n $\Gamma_{dec,B}$= %.3f +- %.3f\n $n_{th}$= %.3f +- %.3f\n$n_{coh}$= %.3f +- %.3f\n$n_{sq}$= %.3f +- %.3f' % (red_chi, fit_rabi, fit_rabi_err, fit_decc, fit_decc_err, fit_decb, fit_decb_err, fit_nth, fit_nth_err, fit_ncoh, fit_ncoh_err, fit_nsq, fit_nsq_err)
        else:
            fit_status = 'fit failed'
        if self.do_plot:
            # print('x-Axis for the blue sideband compressed by a factor of', bluescale)
            a,b,c=blueflop
            blueflop_plt = [bluescale*a,b,c]
            plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [carrflop, blueflop_plt], lbl, fit_status);
            plt.show()

        for (key, val),( _, err),f in zip(m.values.items(), m.errors.items(), fix):
            n = max(significant_digit(err),0)
            s_v = '%%%i.%if'%(6+n,n)
            s_e = '%%%i.%if'%(1+n,n)
            txt_par = '%s\t= %s +- %s'%(key,s_v%val,s_e%err)
            if f>0:
                txt_par += '*'
        return red_chi, fit_valid, value, error


    def single_fit_sb_fock_data(self, redflop, blueflop, lbl, mode_freq, mode_angle, Rabi_init, dec_init=0.001, limb_init=0.55, limr_init=0.85, nth=0.1, ncoh=1e-9, nsq=1e-9, fix=[0,0,0,0,0,1,1], nmax=8, ntrot=1):
        LD = LDparameter(mode_freq,mode_angle)
        print('lamb dicke parameter:',LD)

        init_sb = [Rabi_init,dec_init,limb_init,limr_init]
        red_chi_sb, fmin, param, m, flop_func_list, \
            [fit_rabi, fit_dec, fit_limb, fit_limr], \
            [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err], \
            fit_fockdist_norm, [fock_n, fock_p, fock_e] = \
                fit_flop_sb_fock(redflop, blueflop, LD, nmax, init_sb, fix[0:4])
        fit_sb_valid = fmin['is_valid']

        if self.do_plot:
            if fit_sb_valid:
                #fit_status = 'Fock distr.'
                fit_status = '$red. \chi^2$= %.3f\n $\Omega_{0}$= %.3f +- %.3f\n $\Gamma_{dec}$= %.3f +- %.3f\n$\eta_{LD}$=%.2f' % (red_chi_sb, fit_rabi, fit_rabi_err, fit_dec, fit_dec_err, LD)
            else:
                fit_status = 'Fit failed'
            plot_flop_fit(flop_func_list, fock_n, fock_p, fock_e, [redflop, blueflop], lbl, fit_status, figsize=(8,4));
            plt.show()

        for (key, val),( _, err),f in zip(m.values.items(), m.errors.items(), fix):
            n = max(significant_digit(err),0)
            s_v = '%%%i.%if'%(6+n,n)
            s_e = '%%%i.%if'%(1+n,n)
            txt_par = '%s\t= %s +- %s'%(key,s_v%val,s_e%err)
            if f>0:
                txt_par += '*'
            #print(txt_par)

        init_fock = [nth, ncoh, nsq]
        fix_fock = []
        #red_chi_fock, fmin, param, m, \
        #    [fit_nth, fit_ncoh, fit_nsq], \
        #    [fit_nth_err, fit_ncoh_err, fit_nsq_err], \
        #    fit_fockdist_norm, [fit_fock_n, fit_fock_p] = \
        #        fit_dist_fock(fock_n, fock_p, fock_e, init_fock, fix[4:], ntrot)

        #if self.do_plot:
            #plot_fock_fit(fock_n, fock_p, fock_e, fit_fock_n, fit_fock_p)
            #plt.show()

        #for (key, val),( _, err) in zip(m.values.items(), m.errors.items()):
        #    n = max(significant_digit(err),0)
        #    s_v = '%%%i.%if'%(6+n,n)
        #    s_e = '%%%i.%if'%(1+n,n)
            #print('%s\t= %s +- %s'%(key,s_v%val,s_e%err))

        #fit_fock_valid = fmin['is_valid']
        #value = [fit_rabi, fit_dec, fit_limb, fit_limr, fit_nth, fit_ncoh, fit_nsq]
        value = [fit_rabi, fit_dec, fit_limb, fit_limr, fock_p]
        #error = [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err, fit_nth_err, fit_ncoh_err, fit_nsq_err]
        error = [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err, fock_e]
        #return [red_chi_sb,red_chi_fock], [fit_sb_valid,fit_fock_valid], value, error
        return [red_chi_sb], [fit_sb_valid], value, error


    # from rob, for generalized sideband fit
    def single_fit_sb_carr_fock_data(self, carrflop, blueflop, lbl, bluescale, mode_freq, mode_angle, Rabi_init, decc_init=0.001, decb_init=0.001, limc_init=0.55, limb_init=0.85, nth=0.1, ncoh=1e-9, nsq=1e-9, init_pops=[], fix=[0,0,0,0,0,0,1,1], nmax=8, ntrot=1, n_lhs=0):
        LD = LDparameter(mode_freq,mode_angle)
        print('lamb dicke parameter:',np.round(LD,4))

        init_sb = [Rabi_init,decc_init,decb_init,limc_init,limb_init]
        if len(init_pops)>0:
            print('yo')
            init_sb = np.append(init_sb,init_pops)
            print(init_sb)

        red_chi_sb, fmin, param, m, flop_func_list, values, errors, fit_fockdist_norm, [fock_n, fock_p, fock_e] = \
                fit_flop_carr_bsb_fock(carrflop, blueflop, LD, nmax, init_sb, fix[0:5], bluescale, n_lhs)

        fit_sb_valid = fmin['is_valid']
        [fit_rabi, fit_decc, fit_decb, fit_limc, fit_limb] = values
        [fit_rabi_err,fit_decc_err,fit_decb_err,fit_limc_err,fit_limb_err] = errors

        if self.do_plot:
            if fit_sb_valid:
                fit_status = '$red. \chi^2$= %.3f\n $\Omega_{0}$= %.3f +- %.3f\n $\Gamma_{dec,C}$= %.3f +- %.3f\n $\Gamma_{dec,B}$= %.3f +- %.3f\n$\eta_{LD}$=%.2f' % (red_chi_sb, fit_rabi, fit_rabi_err, fit_decc, fit_decc_err,  fit_decb, fit_decb_err, LD)
            else:
                fit_status = 'Fit failed'

            # print('x-Axis for the blue sideband compressed by a factor of', bluescale)
            a,b,c=blueflop
            plot_flop_fit(flop_func_list, fock_n, fock_p, fock_e, [carrflop, [bluescale*a,b,c]], lbl, fit_status, figsize=(10,5));
            plt.show()

        for (key, val),( _, err),f in zip(m.values.items(), m.errors.items(), fix):
            n = max(significant_digit(err),0)
            s_v = '%%%i.%if'%(6+n,n)
            s_e = '%%%i.%if'%(1+n,n)
            txt_par = '%s\t= %s +- %s'%(key,s_v%val,s_e%err)
            if f>0:
                txt_par += '*'

        init_fock = [nth, ncoh, nsq]
        fix_fock = []

        value = [fit_rabi, fit_decc, fit_decb, fit_limc, fit_limb, fock_p]
        error = [fit_rabi_err,fit_decc_err,fit_decb_err,fit_limc_err,fit_limb_err, fock_e]
        return [red_chi_sb], [fit_sb_valid], value, error



    def single_fit_sb(self, fullpath, *args, **kwargs):
        redflop, blueflop, lbl = open_file(fullpath, self.cache_path)

        fock=False
        if 'fock' in kwargs:
            fock=kwargs['fock']
            del kwargs['fock']

        if fock:
            func = self.single_fit_sb_fock_data
            # hier gehts rein
        else:
            func = self.single_fit_sb_data

        ret = func(redflop, blueflop, lbl, *args, **kwargs)
        self._log('fit %s %s %s %s %i %f %f %f'%('single_fit_sb', '['+','.join(fullpath)+']', 'dummy', func.__name__, fock, \
                    kwargs['mode_freq'], kwargs['mode_angle'], kwargs['Rabi_init']))
        return ret

    # from rob
    def single_fit_sb_carr(self, fullpath, bluescale, *args, **kwargs):
        '''sidebands: [first_sb, second_sb] sidebands used for the fit, if sb>0, start in the dark state, else start in the bright one
            in the following: first_sb = carrier = 0, second_sb = first blue sideband = +1'''
        carrflop, blueflop, lbl = open_file(fullpath, self.cache_path)
        print('first point of carrierflop (should be 1):',carrflop[1][0])
        print('first point of blueflop (should be 0):',blueflop[1][0])

        fock=False
        if 'fock' in kwargs:
            fock=kwargs['fock']
            del kwargs['fock']

        if fock:
            func = self.single_fit_sb_carr_fock_data
            # hier gehts rein
        else:
            func = self.single_fit_sb_carr_data

        ret = func(carrflop, blueflop, lbl, bluescale, *args, **kwargs)
        self._log('fit %s %s %s %s %i %f %f %f'%('single_fit_sb', '['+','.join(fullpath)+']', 'dummy', func.__name__, fock, \
                    kwargs['mode_freq'], kwargs['mode_angle'], kwargs['Rabi_init']))
        return ret

    def multi_sb_fit(self, sb_file_list, inits, fix, mode_freq, mode_angle, nmax = 20, ntrot = 1):
        # sb_file_list = [ [sb0_1, sb0_2], [sb1_1, sb1_2], ..., [sbn_1, sbn_2]]
        n_th_list = []
        n_coh_list = []
        n_sq_list = []
        [Rabi_init,dec_init,limb_init,limr_init, nth, ncoh, nsq] = inits
        for i,fn in enumerate(sb_file_list):
            print('Fit #%i/%i:'%(i+1,len(sb_file_list)))
            _, fit_valid, value, error = \
                self.single_fit_sb(find_files(fn),
                    mode_freq, mode_angle,
                    Rabi_init, dec_init, limb_init, limr_init,
                    nth, ncoh, nsq,
                    fix, nmax, ntrot)
            if np.all(fit_valid):
                n_th = value[4]
                n_th_err = error[4]
                n_coh = value[5]
                n_coh_err = error[5]
                n_sq = value[6]
                n_sq_err = error[6]

                n_th_list.append([n_th, n_th_err])
                n_coh_list.append([n_coh, n_coh_err])
                n_sq_list.append([n_sq, n_sq_err])
            else:
                n_th_list.append([None,None])
                n_coh_list.append([None,None])
                n_sq_list.append([None,None])

        return n_th_list,n_coh_list,n_sq_list

    def single_fit_func(self, data, func_fit, invert, give_all_para=False, name_var='', xlabel=None, ylabel=None, add_contrast=True, text=None, title=None, set_fit=True, plot_residuals=False):
        # fit data
        try:
            x, y, y_err, func_model, peaks, start, popt, perr, R2, chi2, var, var_err = fit_parameter(func_fit, data, invert)
            [*value],[*value_err] = var, var_err

            # plot data & fit
            if self.do_plot:
                f, ax = plot_fit(x,y,y_err,peaks,func_model,start,popt,perr,var,var_err,R2,chi2,invert, lbl=[xlabel, ylabel], plot_residuals=plot_residuals)
                # add label
                #if xlabel is not None:
                #    ax.set_xlabel(xlabel)
                #if ylabel is not None:
                #    ax.set_ylabel(ylabel)
                # add last value
                #if name_var and name_var != 'dummy':
                    #old_value = float(self.get(name_var))
                    #ax.axvline(x=old_value,ls='--',c='grey',label='previous value')
                # add contrast line (from MW flop)
                if add_contrast:
                    mw_contr_high = float(self.get('mw_contr_high'))
                    mw_contr_low = float(self.get('mw_contr_low'))
                    ax.axhline(y=mw_contr_high,ls='--',c='grey',label='upper contrast')
                    ax.axhline(y=mw_contr_low,ls='--',c='grey',label='lower contrast')
                if text is not None:
                    ax.text(1.02, .78, text, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
                if title is not None:
                    f.suptitle(title, fontsize=10)
                plt.show()

                if name_var and name_var != 'dummy' and set_fit:
                    # set parameter from fit
                    if np.isfinite(value_err):
                        self.set_parameter(name_var, value[0], R2)
                    else:
                        print('\t    %s = %s +- %s\t %s\n' % (name_var, str(value), str(value_err), 'fit failed: infinite error'))
        except RuntimeError:
            self.plot_data(data)
            value=None; value_err=None; R2=0; popt=[]; perr=[]; chi2=0.

        # retrun primary or all fit parameter(s)
        if give_all_para:
            return value, value_err, R2, popt, perr, chi2
        else:
            return value, value_err, R2

    def fit_data(self, func, data, start, \
                 plt_labels=['Scan para. (a.u.)','Cts. (a.u.)','data_name','model_func'], \
                 plt_log=[0,0], plot_residuals=False):

        #do the fit
        x,y,y_err = unpack_sorted(data)
        popt, perr, R2, chi2 = fit_func(func, x, y, y_err, start, absolute_sigma=True)
        #do the plot
        f, axs = plot_fit(  x,y,y_err, \
                            peaks=None,func=func,start=start, \
                            popt=popt,perr=perr, \
                            var=[],var_err=[], \
                            gof=R2,chi2=chi2,invert=False, \
                            plot_start=False, plot_patch=False, \
                            plot_residuals=plot_residuals, plot_peaks=False, \
                            plot_label=plt_labels[2:], lbl=plt_labels[:2])

        #plt.xlabel(plt_labels[0])
        if plt_log[0]==1: plt.xscale('log')

        #plt.ylabel(plt_labels[1])
        if plt_log[1]==1: plt.yscale('log')

        #plt.legend(loc=0)
        import datetime
        fname='./data/scans/'+str(datetime.datetime.now())[:22].replace(' ','_').replace(':','-').replace('.','-')+'.png'
        plt.savefig(fname, dpi=125)
        plt.show()
        return popt, perr, R2, chi2

    def prnt_rslts(self, arg_names, popt, perr, verbose=False):
        m,v,e = arg_names, popt, perr
        n = max(significant_digit(e),0)
        #s_v = '%%%i.%if'%(6+n,n)
        #s_e = '%%%i.%if'%(1+n,n)
        #txt += '%5s'%m + ' '+r'= ' + s_v%v + r' $\pm$ ' + s_e%e + '\n'
        if e==float('Inf'):
            s_v = '%%.%if'%(n)
            txt = '%5s'%m + ' '+r'= ' + s_v%v + '(inf)'
        else:
            s_v = '%%.%if'%(n)
            txt = '%5s'%m + ' '+r'= ' + s_v%v + '(%i)'%(e*10**n)
        if verbose==True: print(txt)
        return txt

    def plt_time_series(self, data, lbl=['Duration (s)','RF pwr. (a.u.)']):
        sns.set(font_scale=0.85)
        x, y, yerr = data
        smpl_rate=((x[-1]-x[0])/len(x))**(-1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        txt = 'Total dur. (min): '+str(round((x[-1]-x[0])/60,1))+' / Sample size: '+str(len(y))+''
        txt += '\nDrift: '+ self.prnt_rslts('d/dt', slope*60*1e3, std_err*60*1e3, verbose=False)+'$\,\cdot\,10^{-3}$ min$^{-1}$'
        #time series
        plt.errorbar(x, y, yerr=yerr, linestyle='', marker='o', color='navy', alpha=.5)
        sns.regplot(x, y, color='red')
        #plt.legend()
        plt.title(txt)
        plt.xlabel(lbl[0])
        plt.ylabel(lbl[1])

        #PSD
        plt.axes([.95,.58,.3,.295])
        f, psd = signal.periodogram(y, smpl_rate)
        plt.semilogy(f, psd, color='navy')
        #plt.ylim([1e19*np.min(psd), 10.*np.max(psd)])
        plt.xlabel('Freq. (Hz)')
        plt.ylabel('PSD (a.u.)')
        plt.yticks([])

        #distribution
        plt.axes([.95,0.125,.3,.295])
        sns.distplot(y, kde=False, fit=sci.stats.norm, label='$\Delta$='+str(round(np.std(y),7)), color='navy')
        plt.legend(loc='lower center')
        plt.axvline(x=np.mean(y), color='red')
        plt.yticks([])
        plt.xticks([np.mean(y)])
        plt.xlabel(lbl[1])
        plt.ylabel('Prop.')
        plt.show()
        self.eval_oadev(y, smpl_rate, lbl=lbl[1][:-5], units='', show_ref='None', scl_ref=1, verbose=True, rel=1)

    def plt_longterm(self, ip_l=['fr_1', 'fr_2'], rng=['2019-07-31 09:00:00', '2019-07-31 11:30:00'], verbose=True, rel=1):
        data_l=[]
        for ip in ip_l:
            data=self.read_ip_hstry(ip, rng, verbose=verbose)
            #plt.show()
            if verbose == True:
                self.plt_allan([[ip, data]],rel=rel)
            #plt.savefig('last-allan_'+ip, bbox_inches='tight')
            #plt.show()
            data_l.append([ip, data])
        return data_l

    def plt_allan(self, data_l, rel=1):
        for i in range(len(data_l)):
            data=np.array(data_l[i])[1]
            if len(data)>5:
                ip=data_l[i][0]
                frq_dt=data[:,3]
                smpl_rt=1/np.mean((data[:,0]-data[0,0])/len(data))
                adata=self.eval_oadev(frq_dt, smpl_rt, lbl=ip, units='a.u.', show_ref='None', rel=rel, verbose=True)
                plt.show()
                #return adata
            else:
                print('Not enough data for Allen deviation..')

    def show_log_data(self, ch, wlc):
        from datetime import datetime
        stat, data=wlc.get_trace(ch)
        conf=wlc.get_config()[1][ch]
        t = np.array(data['time'])
        if len(t)>1:
            t0=t[0]
            t_ts = time.localtime(t0)
            t_ts = time.strftime('%Y-%m-%d %H:%M:%S',t_ts)
            t = (t-t0)/60
            if ch != 'WMPres' and ch != 'WMTemp':
                y=np.array(data['trace'])
                y0 = np.array(data['error'])*1e6
                y1 = np.array(data['output'])
            else:
                y0 = np.array(data['trace'])

            fig, ax1 = plt.subplots(figsize=(3., 2.))

            if ch != 'WMPres' and ch != 'WMTemp':
                ax1.plot(t, y0, color='navy')
                ax1.set_xlabel('Log duration (min)')
                ax1.set_ylabel('Deviation (MHz)', color='navy')
                ax1.tick_params('y', colors='navy')

                textst='Activated: '+str(conf['active'])
                textst+='\nLocked: '+str(conf['lock'])
                textst+='\nLast reading (THz):\n'+str(round(data['trace'][-1],6))
                textst+='\nSet (THz):\n'+str(conf['setpoint'])
                textst+='\nLim (V): '+str(np.array(conf['limits'])+conf['offset'])
                ax1.text(1.4, 1.2, textst, transform=ax1.transAxes, fontsize=8,
            verticalalignment='top')
            else:
                ax1.plot(t/60, y0, color='navy')
                ax1.set_xlabel('Log duration (min)')
                if ch == 'WMPres':
                    ax1.set_ylabel('Pressure (hPa)')
                else:
                    ax1.set_ylabel('Temperature (degC)')

            if ch != 'WMPres' and ch != 'WMTemp':
                ax2 = ax1.twinx()
                ax2.plot(t, y1, color='red')
                ax2.set_ylabel('Output (V)', color='red')
                ax2.tick_params('y', colors='red')
                plt.grid(False)
                plt.title(ch+' ('+str(t_ts)+')'+'\nMean (THz): '+str(round(np.mean(y),6)))
            else:
                plt.title(ch+' ('+str(t_ts)+')'+'\nMean: '+str(round(np.mean(y0),1)))
            #fig.tight_layout()
            if len(t)>10: #and ch != 'WMPres' and ch != 'WMTemp':
                            mu, std = norm.fit(y0)
                            xmin=mu-2.5*std
                            xmax=mu+2.5*std
                            x = np.linspace(xmin, xmax, 100)
                            p = norm.pdf(x, mu, std)
                            plt.axes([1.225, .13, .25, .30])
                            plt.hist(y0, bins = int(2*np.log(len(y0))), density=True, color='navy', edgecolor='white')
                            plt.plot(x, p, color='red', linewidth=3, alpha=0.75)
                            plt.axvline(x=0, linewidth=5, color='green', alpha=0.5)
                            plt.xlim((xmin, xmax))
                            plt.xlabel('Freq. dev. (MHz)')
                            plt.ylabel('Prop.')
                            plt.title('std = '+str(round(std, 2))+' MHz')
                            plt.yticks([])
            plt.show()

        else:
            print('No data for: ', ch)
