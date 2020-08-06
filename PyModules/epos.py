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

#from PyModules.analyse_eios.eios_analyse import func_decay_exponential

from PyModules.eios import EIOS_META_SCRIPT
from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save
from PyModules.analyse_eios.eios_sb import LDparameter, fit_flop_sb, fit_flop_sb_fock, fit_dist_fock, fit_flop_carrier, plot_fock_fit, plot_flop_fit, open_file

from PyModules.analyse_eios.eios_analyse import unpack_sorted, significant_digit, plot_fit
from PyModules.analyse_eios.eios_analyse import fit_direct, fit_linear, fit_parameter, fit_func
from PyModules.analyse_eios.eios_analyse import fit_multi_freq, fit_time, fit_phase
from PyModules.analyse_eios.eios_analyse import gauss_sum, lorentz_sum, abs_sum, sinc_sum, parabola_sum, sinc_abs_sum, sincSquare, gauss, lorentz, parabola

from PyModules.utilities import hash

#from PyModules.machine_learning.run_ml import write_waveform
#from PyModules.pdq.waveform import smooth_step_eu, b_ex_1, b_ex_2, d_b_ex_2, linear_ramp

def RandomId(stringLength=6):
    letters = string.ascii_letters + string.digits
    return ''.join([random.choice(letters) for i in range(stringLength)])

# Experiment Python Operating System
class EPOS(EIOS_META_SCRIPT):

    def __init__(self, cache_path='./data/', log_file='./data/log.txt', \
                        gof_thres = .7, do_plot=True, do_live=False, \
                        srv_adr_cmd = '/tmp/socket-eios-cmd', srv_adr_data = '/tmp/socket-eios-data', \
                        wvf_folder='../UserData', wvf_db_file='../UserData/waveform_db.json'):
        super().__init__(srv_adr_cmd=srv_adr_cmd, srv_adr_data=srv_adr_data)

        self.last_script_name = ''
        self.session_id = None

        self.cache_path = cache_path
        self.log_file = log_file
        self.gof_thres = gof_thres

        self.do_plot = do_plot
        self.do_live = do_live

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

    def __del__(self):
        super().__del__()

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
                        ret_value = line_split[4:]
                        if (ret_tstamp>t_start):
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
        from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save
        def show_ips(name, ips):
            _, root = read(name)
            ip = read_xml(root)['ionproperties']
            text_ips='Ion properties:\n\n'
            ip_l = []
            for i in ips:
                text_ips+= str(i) +' = '+str(ip[i])+'\n'
                ip_l.append([str(i),ip[i]])
            return ip_l, text_ips

        def show_script(name):
            _, root = read(name)
            script = read_xml(root)['sourcecode']
            scan_para = read_xml(root)['parameter']['name']
            del_until=script.find('pdq_init(')
            del_from=script.find('read();')
            script=script[del_until:del_from]
            script=script.replace('\n\n\n\n','\n').replace('\n\n\n','\n').replace('\n\n','\n')
            return script, scan_para

        path_data_local = self.results_path
        times=[]
        for i in range(len(run_times)):
            times.append(run_times[i])
        name_l=find_files(times, eios_data_path = path_data_local, year='2020')
        data_l=[]
        script_l=[]
        ips_l=[]
        for l in range(len(name_l)):
            name=name_l[l]
            data=read(name)
            data_l.append(data)
            script, scan_para = show_script(name)
            script_l.append(script)
            ip_l, text_ips=show_ips(name, ips)
            ips_l.append(ip_l)
            raw_data_avg=[]
            for k in range(len(data[0])):
                raw_data_avg.append([data[0][k]['x'],data[0][k]['y'],data[0][k]['error']])
            if verbose==True:
                if len(raw_data_avg)>0:
                    fig, ax = plt.subplots()
                    ColorList=['navy','red','orange','grey','silver','black']
                    if name:
                        fig.canvas.set_window_title(name)
                    for i,data in enumerate(raw_data_avg):
                        x,y,y_err = data
                        plt.errorbar(x, y, yerr = y_err, linestyle = "None", marker = "o", label='Det.# %i'%i, color=ColorList[i], markersize=7.5, lw=1., capsize=.0);
                    plt.legend(loc='upper right')
                plt.title('.'+name[(len(path_data_local)+5):])
                plt.ylabel('Cts.')
                plt.xlabel(scan_para+' (a.u.)')
                plt.text(1.1, .5, 'Experimental sequence:\n'+script_l[l]+'\n'+text_ips, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
                plt.show()
        return name_l, data_l, script_l, ips_l

    def session_start(self):
        if self.session_id is None:
            self.session_id = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
            #self.session_id = RandomId()
            self._log('session %s start' % self.session_id)
            return self.session_id
        return None

    def session_stop(self):
        if self.session_id is not None:
            self._log('session %s stop' % self.session_id)
            tmp_id = self.session_id
            self.session_id = None
            return tmp_id
        return None

    def session_annotate(self, msg):
        if self.session_id is not None:
            if not isinstance(msg, str):
                msg = '%f'%msg
            msg.replace('\n',' ')
            self._log('session %s annotate %s' % (self.session_id, msg))
            return self.session_id
        return None

    def session_get(self, session_id):
        data = []
        stat_stop = False
        stat_start, idx_start, ret_date, ret_time = self.find_log(key='session', name=session_id, value='start')
        if stat_start:
            stat_stop, idx_stop, ret_date, ret_time = self.find_log(key='session', name=session_id, value='stop', timeout=idx_start)
            if stat_stop:
                data = self.get_log_index(idx_start, idx_stop)
        return stat_start, stat_stop, data

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

    def session_replay(self, session_id, key="fit",do_plot=True):
        logs, annotations, data_key = self.session_find(session_id, key)
        ret_list = []
        data_list=[]
        for d_key in data_key:
            script_name = d_key[3]
            #########################
            if script_name=='heating_rates_iteration':
                sids = []
                return d_key[4]
            #########################
            filename = d_key[4]
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
                for i,d in enumerate(data):
                    if i == counter:
                        self.do_plot=do_plot
                        ret = self.single_fit_func([d['x'], d['y'], d['error']], func, invert=invert, \
                                                       give_all_para=True, name_var=name_var, \
                                                       xlabel=None, ylabel=None, add_contrast=True, \
                                                       text=None, title=title, set_fit=False)
                        ret_list.append(ret)
                        #
                        data_list.append(data)
        return ret_list, data_list

    def set(self,name,value):
        old_value = super().get(name)
        ret = super().set(name,value)
        if ret:
            self._log('set %s %s (%s)'%(name, value, old_value))
        return ret

    def set_profile(self,profile_name,shim_name,value):
        old_value = super().get_profile(profile_name,shim_name)
        if super().set_profile(profile_name,shim_name,value):
            self._log('set %s.%s %s (%s)'%(profile_name, shim_name, value, old_value))
            return True
        else:
            return False

    def get_profile(self,profile_name,shim_name):
        last = super().get_profile(profile_name,shim_name)
        last_date = ''
        var_name = '%s.%s'%(profile_name, shim_name)
        succ,_,ret_date,ret_time = self.find_log('set', var_name, value=last)
        if succ:
            last_date = '%s %s'%(ret_date,ret_time)
        return float(last), last_date

    def set_parameter(self, var_name, value, gof=1):
        ret = False
        str_value = str(value)
        if gof>self.gof_thres:
            if np.isfinite(value) and not (value==0):
                last_str = ''
                if var_name:
                    last, last_date = self.get_parameter(var_name)
                    last_str = '[prev. %s (%s)]'%(last, last_date)
                    ret = self.set(var_name, str_value)
                if ret:
                    print('\tset %s = %s %s\t %s\n' % (var_name, str_value, last_str, 'fit OK'))
                    return True
                else:
                    print('\t    %s = %s\t %s\n' % (var_name, str_value, 'fit OK, set failed'))
                    return False
            else:
                print('\t    %s = %s\t %s\n' % (var_name, str_value, 'fit OK, invalid value (infinite/zero)'))
                return False
        else:
            print('\t    %s = %s \t %s\n' % (var_name, str_value, 'fit failed, not set'))
            return False

    def get_parameter(self, var_name):
        last = self.get(var_name)
        last_date = ''
        succ,_,ret_date,ret_time = self.find_log('set', var_name, value=last)
        if succ:
            last_date = '%s %s'%(ret_date,ret_time)
        return float(last), last_date

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

    def plt_live(self, out_box, s_id, data_ctr, last_call=False):
        fig = super().plot_data(name='', data_ctr=data_ctr)
        with out_box:
            clear_output(wait=True)
            display(fig)
            plt.close(fig)
            if last_call:
                clear_output(wait=True)
                #out_box.layout.height = '0px'
                out_box.close()

        return True

    def run_single(self, script_name, par_name=None, \
                        start=None, stop=None, \
                        numofpoints=None, expperpoint=None, rnd_sampled=None, \
                        live_plots_flag=False, counter=0, verbose=True):
        p_name, p_start, p_stop, p_numofpoints, p_expperpoint, p_rnd_sampled, shim_scan, _, _ = super().get_parameter(script_name)
        if par_name is not None:
            p_name = par_name
        if start is not None:
            p_start = start
        if stop is not None:
            p_stop = stop
        if numofpoints is not None:
            p_numofpoints = numofpoints
        if expperpoint is not None:
            p_expperpoint = expperpoint
        if rnd_sampled is not None:
            p_rnd_sampled = rnd_sampled

        if verbose:
            print('Run %s (%s)\n\tstart %f, stop %f, \n\tpoints %u, experiment/point %u rnd %u show %u' % (
                        script_name, p_name,
                        p_start, p_stop,
                        p_numofpoints, p_expperpoint,
                        p_rnd_sampled, live_plots_flag))

        t0 = time.time()
        s_id,name,data = super().run(script_name, p_name,
                                      p_start, p_stop,
                                      p_numofpoints, p_expperpoint,
                                      p_rnd_sampled, live_plots_flag)
        t1 = time.time()
        dt = t1-t0
        name_shrt = name[name.rfind('/')+1:]
        file_hash = hash(name)
        self._log('run %s %s %s %f %f %s'%(script_name, p_name, name_shrt, t0, dt, file_hash))
        if verbose:
            print('\t%s [%s]'%(name_shrt,file_hash[:5]))
            print('\tduration %.2fs'%(dt))

        data_out = []
        if counter is None:
            data_out = data
        elif counter==0:
            data_out = data[0]
        elif counter>0 and counter<len(data):
            data_out = data[int(counter)]

        return name, np.array(data_out)

    def run_stream(self, script_name, par_name=None, \
                        start=None, stop=None, \
                        numofpoints=None, expperpoint=None, rnd_sampled=None, \
                        live_plots_flag=False, counter=0, verbose=True,\
                        block=False, live_plt_call=True):
        p_name, p_start, p_stop, p_numofpoints, p_expperpoint, p_rnd_sampled, shim_scan, _, _ = super().get_parameter(script_name)
        if par_name is not None:
            p_name = par_name
        if start is not None:
            p_start = start
        if stop is not None:
            p_stop = stop
        if numofpoints is not None:
            p_numofpoints = numofpoints
        if expperpoint is not None:
            p_expperpoint = expperpoint
        if rnd_sampled is not None:
            p_rnd_sampled = rnd_sampled

        if verbose:
            print('Run %s (%s) \n\tstart %f, stop %f, \n\tpoints %u, experiments/point %u rnd %u show %u' % (
                    script_name, p_name, p_start, p_stop,
                    p_numofpoints, p_expperpoint,
                    p_rnd_sampled, live_plots_flag))

        if live_plt_call is None or verbose==False:
            live_plt_call= lambda s_id, data_ctr, last_call: True
        elif callable(live_plt_call):
            pass
        else:
            size = rcParams["figure.figsize"]
            dip = rcParams["figure.dpi"]
            height = size[1]*dip
            box_height = int(height*1.25)
            out_box = Output(layout={'height':'%ipx'%box_height})
            #out_box = Output(layout={'flex-grow':'1','flex-shrink':'0'})
            display(out_box)
            live_plt_call = lambda s_id, data_ctr, last_call: self.plt_live(out_box, s_id, data_ctr, last_call)

        t0 = time.time()
        ret = super().add(script_name, live_plt_call,
                                 p_name, p_start, p_stop,
                                 p_numofpoints, p_expperpoint,
                                 p_rnd_sampled, live_plots_flag, block)
        t1 = time.time()
        dt = t1-t0
        s_id,name = ret[0], ret[1]
        name_shrt = name[name.rfind('/')+1:]
        if block:
            file_hash = hash(name)
            self._log('run %s %s %s %f %f %s'%(script_name, p_name, name_shrt, t0, dt, file_hash))
            if verbose:
                print('\t%s [%s]'%(name_shrt,file_hash[:5]))
                print('\tduration %.2fs'%(dt))

            data = []
            if len(ret)>2:
                data = ret[2]
            data_out = []
            if counter is None:
                data_out = data
            elif counter>=0 and counter<len(data):
                data_out = data[int(counter)]
            return name, np.array(data_out)
        else:
            self._log('run %s %s %s %f'%(script_name, p_name, name_shrt, t0))
            if verbose:
                print('\t%s'%(name_shrt))
            return s_id, name

    def run(self, script_name, **kwargs):
        """ Starts a script in EIOS. Required parameter is script_name as string.
            Additional key word parameters are passed on.
            If class variable do_live is True (constructor),
            data is shown while running (run_stream, otherwise run_single).
            Return the name and data of a script run.
        """

        if self.do_live:
            ret = self.run_stream(script_name, block=True, live_plt_call=True, **kwargs)
            name = ret[0]
        else:
            ret = self.run_single(script_name, **kwargs)
            name = ret[0]
        self.last_script_name = name
        return ret

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

    def single_fit_sb_fock_data(self, redflop, blueflop, lbl, mode_freq, mode_angle, Rabi_init, dec_init=0.001, limb_init=0.55, limr_init=0.85, nth=0.1, ncoh=1e-9, nsq=1e-9, fix=[0,0,0,0,0,1,1], nmax=8, ntrot=1, rob=0):
        LD = LDparameter(mode_freq,mode_angle)

        init_sb = [Rabi_init,dec_init,limb_init,limr_init]


        red_chi_sb, fmin, param, m, flop_func_list, \
            [fit_rabi, fit_dec, fit_limb, fit_limr], \
            [fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err], \
            fit_fockdist_norm, [fock_n, fock_p, fock_e] = \
                fit_flop_sb_fock_rob(redflop, blueflop, LD, nmax, init_sb, fix[0:4])

        fit_sb_valid = fmin['is_valid']
        # print("ssss")

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

    def single_fit_sb(self, fullpath, *args, **kwargs):
        redflop, blueflop, lbl = open_file(fullpath, self.cache_path)

        fock=False
        if 'fock' in kwargs:
            fock=kwargs['fock']
            del kwargs['fock']

        if fock:
            func = self.single_fit_sb_fock_data
            print('ddd')
        else:
            func = self.single_fit_sb_data

        ret = func(redflop, blueflop, lbl, *args, **kwargs)
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
                if name_var and name_var != 'dummy':
                    old_value = float(self.get(name_var))
                    ax.axvline(x=old_value,ls='--',c='grey',label='previous value')
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

    def single_fit_run(self, script_name, invert, xlabel=None, ylabel=None, name_var='', add_contrast=True, give_all_para=False, func=None, text=None, **kwargs):
        # select default fit func
        if func is None:
            func = self.fit_single_gauss

        # select counter and set to None -> return all counter (select later)
        counter = 0
        if 'counter' in kwargs:
            counter = kwargs['counter']
        kwargs['counter'] = None

        name, data = self.run(script_name, **kwargs)
        title = '%s%s: %s' % (script_name, '(%s)'%(name_var) if name_var else '', name[name.rfind('/')+1:])

        # make data a list of counter, if only one counter was selected (not the case, since counter always None)
        if len(data.shape)<3:
            data = [data]

        if isinstance(invert, (int, float, bool)):
            invert = [invert]*len(data)
        elif isinstance(invert, list):
            if len(invert) != len(data):
                raise Exception('Size of list "invert" does not match counter number!')
        else:
            raise Exception('Parameter "invert" can only be bool (numeric) or list with same size as counter number!')

        ret_list = []
        for i,(d,inv) in enumerate(zip(data,invert)):
            if i==counter or counter is None:
                ret = self.single_fit_func(d, func, invert=inv, \
                                           give_all_para=give_all_para, name_var=name_var, \
                                           xlabel=xlabel, ylabel=ylabel, add_contrast=add_contrast, \
                                           text=text, title=title)
                ret_list.append(ret)
                self._log('fit %s %s %s %s %i %i %s'%(script_name, name, name_var, func.__name__, inv, i, title))

        if len(ret_list)>1:
            return ret_list
        else:
            return ret

    def single_fit_position(self, name_script, name_var, invert, \
                            xlabel='Position (a.u.)', ylabel='Counts', text=None, **kwargs):
        return self.single_fit_run(script_name=name_script, name_var=name_var, invert=invert, \
                                    xlabel=xlabel, ylabel=ylabel, \
                                    func=self.fit_single_gauss, text=text, **kwargs)

    def single_fit_frequency(self, name_script, name_var, invert, \
                            xlabel='Frequency (MHz)', ylabel='Counts', text=None, **kwargs):
        return self.single_fit_run(script_name=name_script, name_var=name_var, invert=invert, \
                                    xlabel=xlabel, ylabel=ylabel, \
                                    func=self.fit_single_sinc, text=text, **kwargs)

    def single_fit_time(self, name_script, name_var, invert, \
                            xlabel='Time (µs)', ylabel='Counts', text=None, **kwargs):
        return self.single_fit_run(script_name=name_script, name_var=name_var, invert=invert, \
                                    xlabel=xlabel, ylabel=ylabel, \
                                    func=self.fit_time, text=text, **kwargs)

    def single_fit_phase(self, name_script, name_var, invert, \
                            xlabel='Phase (rad)', ylabel='Counts', text=None, **kwargs):
        return self.single_fit_run(script_name=name_script, name_var=name_var, invert=invert, \
                                    xlabel=xlabel, ylabel=ylabel, \
                                    func=self.fit_phase, text=text, **kwargs)

    def single_fit_phase_pi(self, name_script, name_var, invert, \
                            xlabel='Phase (Pi)', ylabel='Counts', text=None, **kwargs):
        return self.single_fit_run(script_name=name_script, name_var=name_var, invert=invert, \
                                    xlabel=xlabel, ylabel=ylabel, \
                                    func=self.fit_phase_pi, text=text, **kwargs)

    def single_fit_linear(self, name_script, name_var, invert, \
                        xlabel='Duration (µs)', ylabel='Avg. thermal <n>', text=None, **kwargs):
        return self.single_fit_run(script_name=name_script, name_var=name_var, invert=invert, \
                                    xlabel=xlabel, ylabel=ylabel, \
                                    func=self.fit_linear, text=text, **kwargs)

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

    # create new waveform from pre-trained model: waveform_%i(wvf_id).json
    def new_waveform(self, name_model, wvf_id, func, do_plot):
        name_wvf = '%s/waveform_%i.json'%(self.wvf_folder, wvf_id)
        time, x, y = write_waveform(func, name_model, name_wvf)
        if do_plot:
            lines = []
            labels = ['Input', 'Output']
            fig, ax1 = plt.subplots()
            lines.append(ax1.plot(time, x[0,:], '.-', color='xkcd:navy')[0])
            ax2=ax1.twinx()
            lines.append(ax2.plot(time, y,'.-', color='xkcd:cyan')[0])
            plt.legend(lines, labels, loc='lower right')
            ax1.set_xlabel('Time [µs]')
            ax1.set_ylabel('RF control [V]')
            ax2.set_ylabel('Pickup PD [V]')
            ax1.yaxis.label.set_color('xkcd:navy')
            ax2.yaxis.label.set_color('xkcd:cyan')
            fig.tight_layout()
            plt.show()
            print('New waveform #%i: %s'%(wvf_id,name_wvf))
        return name_wvf

    # add/edit entry in waveform database
    def add_wvf(self, wvf_id, amp, offset, width=None, comment=''):
        waveform_db = load(self.wvf_db_file)
        key = '%i'%wvf_id
        if key in waveform_db:
            entry = waveform_db[key]
        else:
            entry={}

        entry['id']=wvf_id
        entry['offset']=offset
        entry['amp']=amp
        if width is not None:
            entry['width']=width
        if comment:
            entry['comment']=comment

        waveform_db[key] = entry
        save(self.wvf_db_file, waveform_db)

    # get data from waveform db (data = [wid, offset, amp, width, comment])
    def get_wvf(self, wvf_id):
        waveform_db = load(self.wvf_db_file)
        key = '%i'%wvf_id
        wid = -1
        offset = None
        amp = None
        width = None
        comment = ''
        if key in waveform_db:
            waveform = waveform_db[key]
            wid = waveform['id']
            offset = waveform['offset']
            amp = waveform['amp']
            if 'width' in waveform:
                width = waveform['width']
            if 'comment' in waveform:
                comment = waveform['comment']
        return wid, offset, amp, width, comment

    # set waveform ion property directly
    def set_wvf(self, wvf_id, amp, offset):
        if wvf_id is not None:
            self.set('wf_id',str(wvf_id));
        if amp is not None:
            self.set('amp',str(amp));
        if offset is not None:
            self.set('offset',str(offset));

    # select waveform from db and set ion property
    def select_waveform(self, wvf_id, amp=None):
        wid, offset, _, width, comment = self.get_wvf(wvf_id)
        if wid==wvf_id:
            amp_str = ''
            if amp is not None:
                amp_str = ', amp %.3f'%amp
            self.set_wvf(wid, amp, offset)
            print('select wvf #%i: width %.2f µs (offset %.3f%s)'%(wid,width,offset,amp_str))
            return True
        else:
            return False

    # create new waveform with gaussian shape
    def new_gauss_waveform(self, name_model, wvf_id, A, dip_depth, t_width, do_plot=False):
        func = lambda t,t0: b_ex_2(t,A,dip_depth,t_width,t0)
        self.add_wvf(wvf_id, 1.0, 0., t_width, comment='A=%f, depth=%f'%(A,dip_depth))
        return self.new_waveform(name_model, wvf_id, func, do_plot)

    # create new waveform, select it and run script
    def run_pulse(self, name_script, amp, offset, name_model, wvf_id=99, A=656.2e-3, dip_depth = 656.2e-3-710.0e-3, t_width=.5, do_plot=False, **kwargs):
        self.new_gauss_waveform(name_model, wvf_id=wvf_id, A=A, dip_depth=dip_depth, t_width=t_width, do_plot=do_plot);
        self.set_wvf(wvf_id, amp, offset);
        return self.run_stream(name_script, **kwargs)

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

    def read_avg_adc_me(self, ch=30, no_avg=100, verbose=False):
        avg_adc_val=[]
        for i in range(no_avg):
            avg_adc_val.append(self.read_adc_me(ch))
        m, err = np.mean(avg_adc_val), np.std(avg_adc_val)/np.sqrt(no_avg)
        self.prnt_rslts('Ch'+str(ch), m, err, verbose=verbose)
        return m, err

    def opt_shim_data(self, script_name, shim, delta, profile_name='EU_cool', **kwargs):
        shim_preset, last_date = self.get_profile(profile_name=profile_name, shim_name=shim)
        E, E_err, R2, popt, perr, chi2 = self.single_fit_position(script_name, 'dummy', \
                                                    invert=False, give_all_para=True, \
                                                    xlabel='Shim voltage [a.u.]', ylabel='Counts', \
                                                    start=shim_preset-delta/2, stop=shim_preset+delta/2, **kwargs)
        return E[0], E_err[0], R2, popt, perr, chi2, shim_preset, last_date

    def opt_mirror(self, script, name_var, mirror_idx, delta, **kwargs):
        pos_start = float(self.get(name_var))
        pos, pos_err, gof = self.single_fit_position(script, name_var, start=pos_start-delta/2, stop=pos_start+delta/2, **kwargs)
        if gof>self.gof_thres:
            if self.mirror_pos(mirror_idx, pos[0]):
                print('Mirror #%i position: %f +- %f'%(mirror_idx,pos[0],pos_err[0]))
        else:
            if self.mirror_pos(mirror_idx,pos_start):
                print('Mirror #%i position: %f reset!'%(mirror_idx,pos_start))
        return pos[0],pos_err[0],gof

    def opt_dac(self, script, name_var, dac_idx, delta, **kwargs):
        pos_init = self.get_dac_me(dac_idx)
        pos_start = float(self.get(name_var))
        pos,pos_err,gof = self.single_fit_position(script, name_var, start=pos_start-delta/2, stop=pos_start+delta/2, **kwargs)
        if gof>self.gof_thres:
            if self.set_dac_me(dac_idx,pos[0]):
                print('Mirror #%i position: %f +- %f'%(dac_idx,pos[0],pos_err[0]))
        else:
            self.set_dac_me(dac_idx,pos_init)
        return pos[0],pos_err[0],gof

    def cal_dur(self, script, var_name, invert=False, rng=3.75, \
                expperpoint=50, numofpoints=35, verbose=False, **kwargs):
        t_pi = float(self.get(var_name))
        t0 = 0.
        t1 = rng*t_pi
        return self.single_fit_time(script, var_name, invert=invert, \
                                    expperpoint=expperpoint, numofpoints=numofpoints, \
                                    start=t0, stop=t1, verbose=verbose, **kwargs)

    def cal_fr(self, script, var_name, dur, invert=True, rng=2.5, \
               expperpoint=50, numofpoints=35, verbose=False, **kwargs):
        fr = float(self.get(var_name))
        width = rng/dur
        fr_0 = fr - width/2.
        fr_1 = fr + width/2.
        return self.single_fit_frequency(script, var_name, invert=invert, \
                                                    expperpoint=expperpoint, numofpoints=numofpoints, \
                                                    start=fr_0, stop=fr_1, verbose=verbose, **kwargs)

    def print_header(self, txt):
        self.session_start();
        self.session_annotate(txt);

        cal_begin=datetime.datetime.now()
        print('----------------------------------')
        print(txt)
        print(str(cal_begin))
        print('Please wait...')
        print('---------------------------------- \n')
        return cal_begin

    def print_footer(self, cal_begin):
        sid = self.session_stop()
        print('End session: %s'%sid)

        cal_end = datetime.datetime.now()
        cal_dur = cal_end-cal_begin
        print('----------------------------------')
        print('Done. Seq. took (hours:min:sec):')
        print(str(cal_dur))
        print('---------------------------------- \n')
        return cal_dur
