#!/usr/bin/env python3

import datetime
import numpy as np
import time
import math
import pandas as pd
import random

from threading import Thread
from PyModules.wavemeter.pid_wrapper import pid_container
from PyModules.wavemeter.lock_client import web_lock_client
    
from PyModules.epos import EPOS
from PyModules.utilities import do_async, integer_hill_climb, wait

from PyModules.analyse_eios.eios_analyse import significant_digit, round_sig
from PyModules.analyse_eios.eios_analyse import func_decay_exponential, func_decay_reciprocal, sincSquare

import PyModules.analyse_eios.eios_data as eios_file
from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save
from PyModules.pdq.pdq_waveform import save_wvf

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy import signal, stats
from scipy.stats import norm

from IPython.display import clear_output, display, HTML

class EURO(EPOS):

    def __init__(self, N=35, Nexpp_coarse=50, Nexpp_fine=200, counter_level=[3,9,14,20], 
                    results_path='/home/bermuda/Results_H/tiamo3.sync', 
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
    
    
    def PID_rf(self, set_val=0.071, status=1):
        if status==1:
            self.set('set_RF_pwr',str(set_val))
            offset,_=self.get_shim('RFMod')
            self.runit=True 
            self.pid = pid_container(self.func_read, self.func_write, lock=False, offset=offset, P=25, I=30, D=0, setpoint=set_val, limits=[-0.25,0.25], tracelen=1000)
            self.th = Thread(target = self.run_lock, args=(self.pid,1))
            self.th.start()
            self.pid.set_lock(True)
            return self.pid, self.th

        if status==0 and self.runit==True:
            self.stop_lock()
            return self.pid, self.th
        
    def func_read(self, channel=31, no_avg=1000):
        pwr_rf_l=[]
        for i in range(no_avg):
            pwr_rf_l.append(self.read_adc_me(channel))
        pwr_rf=np.mean(pwr_rf_l)
        return pwr_rf

    def func_write(self, value, last_value):
        #self.set_shim('RFMod',str(value), verbose=False)
        self.set_profile(profile_name='std_cool',shim_name='RFMod',value=str(value))
        self.check_ion(verbose=False)
        return value

    def run_lock(self, sampling):
        self.runit=True
        starttime = time.time()
        wait = sampling
        while self.runit:
            now, val, _ = self.pid()
            t_wait =  wait - np.fmod(float(now-starttime),sampling)
            time.sleep(t_wait)

    def stop_lock(self):
        self.runit=False
        self.th.join()

    def plt_lock_data(self, x, y):
        set_val=self.get('set_RF_pwr')
        set_val=float(set_val)
        #x, y = self.pid.get_trace()
        txt = 'RF lock:'
        txt += '\n'+ self.prnt_rslts('Mean dev', set_val-np.mean(y), np.std(y))
        plt.axes([.0, .12, .72, .76])
        plt.plot(x-np.array(x)[0], y, 
                         ls='--', marker = 'o', markersize=5., color='navy')
        sns.regplot(x-np.array(x)[0], y, color='red')
        plt.hlines(set_val, -5, 1.1*np.max(x-np.array(x)[0]), color='green', linewidth=2.)
        plt.xlabel('Duration (s)')
        plt.ylabel('RF pwr (a.u.)')
        plt.title(txt)
        if len(x)>10:
            #sampl_rate = ((x-np.array(x)[0])/len(x))**(-1)
            f, Pxx_den = signal.periodogram(y, 1)
            mu, std = norm.fit(y)
            
            #print(slope, intercept, r_value, p_value, std_err)
            xmin=mu-2.5*std
            xmax=mu+2.5*std
            #n=max(significant_digit(std),0)
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.axes([.77, .12, .25, .30])
            plt.hist(y, bins = int(len(y)/(len(x)/10)), density=True, color='navy', edgecolor='white')
            plt.plot(x, p, color='red', linewidth=2, alpha=0.75)
            plt.xlim((xmin, xmax))
            #plt.xlabel(ip+' (a.u.)')
            plt.ylabel('Prop.')
            #plt.xticks([round(mu-1.5*std,n), round(mu+1.5*std,n)])
            plt.yticks([])
            #plt.grid(False)
            #plt.yticks([])
            #plt.grid(False)
            plt.axes([.77, .58, .25, .30])
            plt.semilogy(f, Pxx_den, color='navy')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (a.u.)')
            plt.yticks([])
            #plt.grid(False)
            plt.ylim(1e-3*np.mean(Pxx_den), 2*np.max(Pxx_den))
        plt.show()

    def plt_raw_dt(self, data, name, xlabel='Scan_par (a.u.)', ylabel='Cts.', text=''):
            high_fluo=float(self.get('mw_contr_high'))
            low_fluo=float(self.get('mw_contr_low'))
            x = data[0]
            y = data[1]
            y_err = data[2]
            fig, ax = plt.subplots()
            plt.errorbar(x, y, y_err, marker = 'o', markersize=3., lw=1, ls='',fmt='', capsize=2, color='navy')
            plt.title(name[-23:])
            if text is not None:
                    ax.text(1.02, .78, text, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
            plt.axhline(high_fluo, color='gray', ls='--')
            plt.axhline(low_fluo, color='gray', ls='--')

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
        
    def create_noise_burst_wvf(self, noise_dur_mu=1, amp=1., verbose=True):
        noise_dur=int(noise_dur_mu)*1e1*10*1e-8
        npts=int(noise_dur/10*1e8)
        

        wvf=[]
        for i in range(npts):
            wvf.append([i/npts*noise_dur,2*amp*(random.random()-1/2),0.0])
        #wvf.append([(npts+1)/npts*noise_dur,0.0,0.0])
        wvf=np.array(wvf)
        if verbose:
            print('Burst duration (µs):',noise_dur*1e6, 
              '\nNo. of wvf pts:',npts, 
              '\nAmp. (V):',amp)
            if amp!=0:
                self.plt_time_series([wvf[:,0], wvf[:,1], wvf[:,2]], lbl=['Duration (s)','PDQ wvf (V)'])

        x = np.array([wvf[:,1]])
        times = wvf[:,0]

        data_wf = x.reshape((x.shape[0],1,x.shape[1]))
        data_wf = np.insert(data_wf, data_wf.shape[0], x[0,], axis=0)
        #print(data_wf)
        data_file='../UserData/waveform_'+str(int(amp))+'.json'

        save_wvf(times, data_wf=data_wf, data_file=data_file)
        if verbose:
            print('Written to file:', data_file)
        self.add_wvf(wvf_id=int(amp), amp=amp, offset=0.0, width=noise_dur*1e6, comment='White noise (+/-'+str(int(amp))+'V / 4 MHz)')
        return amp, noise_dur*1e6, [wvf[:,0], wvf[:,1]], data_file
    
    def create_eios_script(self, seq = ['PDQ_init','COOL_d', 'DET_bdx'], verbose=True, s_hlp=False):
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
        
    def check_ion(self, verbose=True, numofpoints=0, expperpoint=250):
        _,data=self.run('BDX', numofpoints=numofpoints, expperpoint=expperpoint, verbose=False)
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
        self.ttl_ME(ch_pmt,state)

    def open_photo_laser(self, state, ch_load=7):
        self.ttl_ME(ch_load,state)
        
    def heat_oven(self, state, ch_oven_0=4, ch_oven_1=5):
        self.ttl_ME(ch_oven_1,0)
        self.ttl_ME(ch_oven_0,state)
        time.sleep(1.)
        
    def pulse_abl_laser(self, Npulses):
        #self.ttl_ME(7,1)
        #time.sleep(.5)
        for i in range(Npulses):
            self.ttl_ME(3,1)
            time.sleep(.1) #0.018
            self.ttl_ME(3,0)
            time.sleep(.1) #0.018
        #self.ttl_ME(7,0)
        time.sleep(.1)
        
    def drop_ions(self):
        print('Dropping ions!')
        self.set_shim('RFMod','0.05', verbose=False)
        self.check_ion(verbose=False)
        self.set_shim('RFMod','3.5', verbose=False)
        self.check_ion(verbose=False)
    
    def keep_one_ion(self):
        self.set_shim('RFMod','3.5',verbose=False)
        self.check_ion(verbose=False);
        time.sleep(0.3)
        self.set_shim('RFMod','1.2',verbose=False)
        self.check_ion(verbose=False);
        time.sleep(0.3)
        self.set_shim('RFMod','3.5',verbose=False)
        self.check_ion(verbose=False);
        print('Maybe one ion left')
        
    def mod_RFdrive(self, state, ch_rf=2):
        self.ttl_ME(ch_rf,state)
        #time.sleep(1.)

    def get_all_ips(self, loc=None):
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
            self.set('pwr_rf',str(pwr_rf))
            return pwr_rf
        else:
            pwr_rf, err=self.read_avg_adc_me(ch=31, no_avg=NoAvg, verbose=False)
            self.set('pwr_rf',str(pwr_rf))
            return pwr_rf, err
        
    def get_rf_pwr_in(self, NoAvg=10):
        if NoAvg == 1:
            pwr_rf=self.read_adc_me(30)
            self.set('pwr_rf_in',str(pwr_rf))
            return pwr_rf
        else:
            pwr_rf, err=self.read_avg_adc_me(ch=30, no_avg=NoAvg, verbose=False)
            self.set('pwr_rf_in',str(pwr_rf))
            return pwr_rf, err
        
    def errplt(self, res, lbls=['Scan para (a.u.)','Results (a.u.)'], no=0):
        npres=np.array(res)
        plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.0125)
        ax=plt.subplot(221)
        plt.errorbar(npres[:,0], npres[:,1], yerr=npres[:,2], label='', marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)
        plt.grid(True)
        plt.xlabel(lbls[0])
        plt.ylabel(lbls[1])
        plt.time
        plt.show()   
        #plt.savefig('last', bbox_inches='tight')
    
    def load_ions(self, N, t_heat=150, timeout=180, OnCam=1):
        cal_begin=self.print_header('Loading procedure')
        print('Try loading %i ion(s)'%N)
        if OnCam==0:
            print('Switch to pmt')
            self.switch_to_pmt(True)
        self.open_photo_laser(False)
        cnt, fluo = self.count_ions(verbose=False, give_all_param=True)

        html_str = '<p style="font-family: monospace, monospace;"><b>&emsp;%s</b></p>'
        disp_txt = 'Counted %i ion(s) (BDX cnt %.2f)'%(cnt,fluo)
        dh = display(HTML(html_str%disp_txt),display_id=True)

        if cnt==N:
            print('Already (%i) ion(s) loaded!'%N)
            return cnt==N
        try:
            pht_stat=0.0#self.qck_lck_photo(freq=525.40565, lck_dur=0.2)
            self.heat_oven(True)
            #self.run('load_ions', numofpoints=0, expperpoint=1, verbose=False)
            print('Oven on: pre-heat')
            wait(t_heat)
            
            t0 = time.time()
            i = 0
            while(True):
                i+=1
                tn = time.time()
                dt = tn-t0
                
                if (dt)>timeout:
                    print('Timeout (%.1f sec)'%timeout)
                    print('Check:')
                    print('\t Laser: BD')
                    print('\t Laser: Photo (%.5f THz) (blocked?)'%(525.40545))
                    print('\t PDQ state (restart?)')
                    print('\t RF signal (RS at 16.0 dBm) on?')
                    break
                #print('Try %2i: %.2fs '%(i,dt),end='')
                pht_stat=0.0#self.qck_lck_photo(freq=525.40565, lck_dur=.25)
                data = self.run('BDX', numofpoints=0, expperpoint=1, verbose=False)
                #data = self.run('load_ions', numofpoints=0, expperpoint=1, verbose=False)
                self.open_photo_laser(True)
                if N==1: 
                    time.sleep(2.)
                else:
                    time.sleep(3.5)
                self.open_photo_laser(False)
                
                time.sleep(.05)
                cnt, fluo = self.count_ions(verbose=False, give_all_param=True)
                disp_txt = 'Try %3i: %7.1fs (Stray light %4.1f): counted %i ion(s) (BDX %4.1f) %s'%(i,dt,data[1][-1],cnt,fluo, pht_stat)
                #disp_txt += pht_stat
                dh.update(HTML(html_str%disp_txt))

                if cnt == N:
                    print('Successfully loaded %i ions!'%cnt)
                    self.set('ion_loaded',str(time.time()));
                    break
                elif cnt>N:
                    print('Too many ions loaded (%i)! Discard? Continue...? '%cnt)
                    self.drop_ions()
            print('Oven off!?')
            self.heat_oven(False)
        except:
            cnt=-1
            self.heat_oven(False)
        print('Loading took (h, min, sec): ', datetime.datetime.now()-cal_begin)        
        return cnt==N
    
    def set_shim(self, shim, value, verbose=True):
        prev, date=self.get_profile(profile_name='std_cool',shim_name=shim)
        self.set_profile(profile_name='std_cool',shim_name=shim,value=str(value))
        self.set_profile(profile_name='std_cool_low',shim_name=shim,value=str(value))
        if verbose==True:
            print('set '+shim+'(std_cool/std_cool_low) = '+str(value)+'  [prev. '+str(prev)+' ('+date+')]	 fit OK\n')
        
    def get_shim(self, shim_name, profile_name='std_cool'):
        return self.get_profile(profile_name=profile_name, shim_name=shim_name)
    
    def print_shim_settings(self, profile_name='std_cool'):
        def print_shim_list(shim_list,profile_name,unit=''):
            n = np.max([len(s) for s in shim_list])
            msg = r'%s '+'(%s): '%(unit)+r'%10f [%s]'
            for shim_name in shim_list:
                value, value_date = self.get_profile(profile_name=profile_name,shim_name=shim_name)
                print(msg%(shim_name.ljust(n), value, value_date))
        print('--------------------------')
        print('Profile: ', profile_name)
        print('--------------------------')
        RFMod, RFMod_date = self.get_profile(profile_name=profile_name,shim_name='RFMod')
        print('RF pwr control / RFMod (V): %.3f [%s]'%(RFMod, RFMod_date))
        print_shim_list(['Mesh'],profile_name,r'V')
        print('--------------------------')
        print_shim_list(['Ex', 'Ey', 'Ez'],profile_name,r'kV/m')
        print('--------------------------')
        print_shim_list(['Hxx', 'Hzz', 'Hxy', 'Hxz', 'Hyz'],profile_name,'MHz^2')
        print('--------------------------')
        print_shim_list(['ZCoilCurrent'],profile_name,'V')
        print('')
    
    def print_mw_prop(self):
        fr_mw_3p3_2p2, fr_mw_3p3_2p2_date = self.get_parameter('fr_mw_3p3_2p2')
        fr_mw_3p1_2p2, fr_mw_3p1_2p2_date = self.get_parameter('fr_mw_3p1_2p2')
        fr_mw_3p1_2p0, fr_mw_3p1_2p0_date = self.get_parameter('fr_mw_3p1_2p0')
        print('--------------------------')
        print('Hyperfine transitions freq. (2pi MHz)')
        print('--------------------------')
        print('3p3 - 2p2: %.3f [%s]'%(fr_mw_3p3_2p2,fr_mw_3p3_2p2_date))
        print('3p1 - 2p2: %.3f [%s]'%(fr_mw_3p1_2p2,fr_mw_3p1_2p2_date))
        print('3p1 - 2p0: %.3f [%s]'%(fr_mw_3p1_2p0,fr_mw_3p1_2p0_date))
        t_mw_3p3_2p2, t_mw_3p3_2p2_date = self.get_parameter('t_mw_3p3_2p2')
        t_mw_3p1_2p2, t_mw_3p1_2p2_date = self.get_parameter('t_mw_3p1_2p2')
        t_mw_3p1_2p0, t_mw_3p1_2p0_date = self.get_parameter('t_mw_3p1_2p0')
        print('--------------------------')
        print('Pi-time (µs)')
        print('--------------------------')
        print('3p3 - 2p2: %.3f [%s]'%(t_mw_3p3_2p2,t_mw_3p3_2p2_date))
        print('3p1 - 2p2: %.3f [%s]'%(t_mw_3p1_2p2,t_mw_3p1_2p2_date))
        print('3p1 - 2p0: %.3f [%s]'%(t_mw_3p1_2p0,t_mw_3p1_2p0_date))
        
        print('')    
    
    def print_mode_prop(self, Nion=1):
        fr_1, fr_1_date = self.get_parameter('fr_1')
        fr_2, fr_2_date = self.get_parameter('fr_2')
        fr_3, fr_3_date = self.get_parameter('fr_3')
        if Nion==1:
            print('--------------------------')
            print('Single-Ion Freq. (2pi MHz)')
            print('--------------------------')
            print('COM-1: %.4f [%s]'%(fr_1,fr_1_date))
            print('COM-2: %.4f [%s]'%(fr_2,fr_2_date))
            print('COM-3 : %.4f [%s]'%(fr_3,fr_3_date))
            print('--------------------------')
            print('Tr (MHz^2):\t'+str(round(fr_1**2+fr_2**2+fr_3**2,2)))
            print('\n')
            return fr_1, fr_2, fr_3
        
    def print_raman_prop(self):
        t_b1, t_b1_date = self.get_parameter('t_b1')
        t_b2, t_b2_date = self.get_parameter('t_b2')
        t_r2, t_r2_date = self.get_parameter('t_r2')
        
        print('--------------------------')
        print('Raman AC Stark shifts (kHz)')
        print('--------------------------')
        print('B1: %.1f [%s]'%(500/t_b1, t_b1_date))
        print('B2: %.1f [%s]'%(500/t_b2, t_b2_date))
        print('R2: %.1f [%s]'%(500/t_r2, t_r2_date))
        print('')
    
    def ion_report(self, profile_name='std_cool', Nion=1):
        self.print_header('Ion Report');
        if self.count_ions()>0:
            timestamp=float(self.get('ion_loaded'))
            now=datetime.datetime.utcfromtimestamp(time.time())
            timestr = datetime.datetime.utcfromtimestamp(timestamp)#.strftime('%Y-%m-%d %H:%M:%S')
            print('Ion(s) lifetime:', (now-timestr))
        else:
            print('No ion(s) trapped!?')
        print('BD laser (MHz): '+self.get('fr_laser_bd'))
        self.print_shim_settings(profile_name)
        self.print_mode_prop(Nion)
        self.print_raman_prop()
        self.print_mw_prop()  
    
    def qck_lck_photo(self, freq=525.40575, lck_dur=2., verbose=False):
        wlc = web_lock_client(host='10.5.78.145', port=8000)
        BD_wmch = 'WMCH1'
        Photo_wmch = 'WMCH3'
        wlc.set_setpoint(Photo_wmch, freq)
        wlc.deactivate(BD_wmch)
        wlc.activate(Photo_wmch)
        wlc.lock(Photo_wmch)
        time.sleep(lck_dur)
        gt_trc_Photo=wlc.get_trace_last(Photo_wmch)
        wlc.deactivate(Photo_wmch)
        #wlc.unlock(Photo_wmch)
        wlc.activate(BD_wmch)
        #wlc.lock(BD_wmch)
        #print(gt_trc_Photo)
        str_ret='Photo dev. (MHz):'+str(round((freq-gt_trc_Photo[1]['trace'])*1e6,1))
        if verbose == True:
            print(str_ret)
        return str_ret
    
    def set_RD_freq(self, cnt=0.62, delta=0.125, npts=10):
        res=[]
        for tdac in np.linspace(cnt-delta/2,cnt+delta/2, npts):
            self.set_dac_me(0,tdac)
            name,dataRD=self.run('RD_Pump_opt', numofpoints=0, verbose=False)
            #name,dataRP=euro.run('RP_Pump_opt', numofpoints=0, verbose=False)
            #res.append([tdac,dataRD[1][0],dataRD[2][0],dataRP[1][0],dataRP[2][0]])
            res.append([tdac,dataRD[1][0],dataRD[2][0]])
        data=np.array(res)
        val,_,_=self.single_fit_func([data[:,0],data[:,1],data[:,2]], self.fit_single_gauss, invert=False, xlabel='RD freq. (V)', ylabel='Fluo. (cts)')
        self.set_dac_me(0,val)
        return val
        
    
    def set_BDx_freq(self, freq=536.042513, nexp=250):
        wlc = web_lock_client(host='10.5.78.145', port=8000)
        BD_wmch = 'WMCH1'
        Photo_wmch = 'WMCH3'
        wlc.activate(BD_wmch)
        wlc.deactivate(Photo_wmch)
        wlc.set_setpoint(BD_wmch, freq)
        wlc.lock(BD_wmch)
        self.set('fr_laser_bd',str(freq*1e6))
        time.sleep(2)
        stat, data = wlc.get_trace(BD_wmch)
        wave_b=np.array(data['trace'])[-1]
        _, fluo=self.run('BDX', numofpoints=0, expperpoint=nexp, verbose=False)
        stat, data = wlc.get_trace(BD_wmch)
        wave_a=np.array(data['trace'])[-1]
        wave=np.mean([wave_a,wave_b])
        wave_err=np.std([wave_a,wave_b])
        self.prnt_rslts('BDx_freq', wave, wave_err, verbose=True)
        self.prnt_rslts('Cts', fluo[1][0], fluo[2][0], verbose=True)
        return wave, wave_err, fluo[1][0], fluo[2][0]
    
    def cal_BDx_freq(self, delta_vis=[-20,5], npts=7, nexp=500):
        wlc = web_lock_client(host='10.5.78.145', port=8000)
        BD_wmch = 'WMCH1'
        Photo_wmch = 'WMCH3'
        
        offset=float(self.get('fr_laser_bd'))*1e-6
        wave_rng=offset+np.array(delta_vis)*1e-6
        wlc.set_setpoint(BD_wmch, wave_rng[0])
        time.sleep(2.)
        res=[]
        for i in np.linspace(wave_rng[0],wave_rng[1],npts):
            wlc.set_setpoint(BD_wmch, i)
            time.sleep(1.)
            stat,data = wlc.get_trace(BD_wmch)
            wave_b=np.array(data['trace'])[-1]
            _, fluo=self.run('BDX', numofpoints=0, expperpoint=nexp, verbose=False)
            stat,data = wlc.get_trace(BD_wmch)
            wave_a=np.array(data['trace'])[-1]
            wave=np.mean([wave_a,wave_b])
            res.append([2*(wave*1e6-offset*1e+6), fluo[1][-1], fluo[2][-1]])
            data=np.array(res)
            self.plot_data([data[:,0],data[:,1],data[:,2]], lbls=['BDx freq(MHz)','Fluo. cts.'])
            plt.show()
            
            clear_output(wait=True)

        data=np.array(res)
        #val, err, gof=self.single_fit_func([2*(data[:,0]-offset*1e+6),data[:,1],data[:,2]],
                                           #self.fit_single_lorentz, invert=False, 
                                          # xlabel='BD freq. (MHz)', ylabel='Fluo. (cts)')
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fname = 'bdx'+str(timestr)+'.txt'
        print(fname)
        np.savetxt(fname,data,delimiter=',',header='Freq,cnts,yerr',footer=str(timestr))
        self.plot_data([data[:,0],data[:,1],data[:,2]], name=fname, lbls=['BDx freq(MHz)','Fluo. cts.'])
        plt.show()
        #if gof>.75:
        #    wlc.set_setpoint(BD_wmch, val*1e-6)
        #    self.set('fr_laser_bd',str(val))
            #wlc.set_setpoint(BD_wmch, wave_rng[0])
        #else:
        wlc.set_setpoint(BD_wmch, offset)

    def cal_modefr(self, script, mode_name, amp, dur, **kwargs):
        self.set('t_tickle',str(dur))
        self.set('t_tickle_'+mode_name,str(dur));

        self.set('u_tickle',str(amp))
        self.set('u_tickle_'+mode_name,str(amp));

        el_num = self.get('shim_tickle_' + mode_name);
        self.set('shim_tickle', el_num);

        return self.cal_fr(script, 'fr_'+mode_name, dur, invert=True, give_all_para=True, **kwargs)
    
    def stab_modefreq_rf(self, mode_name, amp, dur, lock=1, expperpoint=150, numofpoints=4, rng=.05, verbose=True, **kwargs):
        fr=float(self.get('fr_'+mode_name))
        self.set('fr_tickle', str(fr))
        self.set('t_tickle',str(dur))
        self.set('t_tickle_'+mode_name,str(dur));
        self.set('u_tickle',str(amp))
        self.set('u_tickle_'+mode_name,str(amp));
        Urf,_=self.get_shim('RFMod')
        start=Urf-rng
        stop=Urf+rng
        el_num=self.get('shim_tickle_'+mode_name);
        self.set('shim_tickle', el_num);
        val, err, R2, popt, perr, chi2 = 0,0,0,0,0,0
        if self.count_ions()>0:
            val, err, R2, popt, perr, chi2 = self.single_fit_position('PDQ_Tickle_BDX1', 'op_RFMod', invert=True, par_name='std_cool.RFMod', expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, live_plots_flag=False, give_all_para=True, verbose=verbose, **kwargs)
            if R2>0.75 and lock==1:
                self.set_shim('RFMod',str(val), verbose=verbose)
        return val, err, R2, popt, perr, chi2
    
    def tickle_ramsey(self, script, mode_name, amp, dur, twait, expperpoint=250, numofpoints=11, rnd_sampled=False, **kwargs):
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
        el_num=self.get('shim_tickle_'+mode_name);
        self.set('shim_tickle', el_num);
        popt, perr = [0,0,0,0,0], [0,0,0,0,0]
        if self.count_ions()>0:
            val, err, R2, popt, perr, chi2 = self.single_fit_phase_pi(script, 'dummy', invert=True, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, live_plots_flag=False,  give_all_para=True, rnd_sampled=rnd_sampled, **kwargs)
        return popt, perr, R2

    def get_mode_coh(self, mode_name, amp, dur, Ramsey_durs, expperpoint=250, numofpoints=11, rnd_sampled=False, start=[1e4, 2]):
        cal_begin=self.print_header('Mode #'+mode_name+' coherence')
        res=np.array([])
        self.cal_modefr('PDQ_Tickle_BDX1', mode_name, amp, dur, \
                        expperpoint=expperpoint, numofpoints=14, rng=2.5)
        n=0
        for i in Ramsey_durs:
            if self.count_ions()>0:
                print('------------------------------')
                print('Ramsey duration (µs): ',i)
                print('------------------------------')
                self.cal_modefr('PDQ_Tickle_BDX1', mode_name, amp, dur, \
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
    
    def get_mode_stab(self, mode_name, amp, durs, expperpoint=250, numofpoints=11, rnd_sampled=False):
        cal_begin=self.print_header('Mode #'+mode_name+' stability')
        res=np.array([])
        n=0
        res=np.array([])
        resQ=np.array([])
        for i in durs:
            if self.count_ions()>0:
                print('------------------------------')
                print('Ramsey duration (µs): ',i)
                print('------------------------------')
                Tamp=amp*durs[0]/i
                value, value_err, R2, popt, perr, chi2=self.cal_modefr('PDQ_Tickle_BDX1', mode_name, Tamp, i, expperpoint=expperpoint, numofpoints=numofpoints, rng=3, rnd_sampled=rnd_sampled)
                if R2>0.5:
                    n=n+1
                    res=np.append(res,(float(i), value, value_err))
                    resQ=np.append(resQ,(float(i), popt[2], perr[2]))
                    rres=res.reshape((n,3))
                    rresQ=resQ.reshape((n,3))
                    data=[rres[:,0], rres[:,1], rres[:,2]]
                    dataQ=[rresQ[:,0], rresQ[:,1], rresQ[:,2]]
                self.print_header('Check mode stability')
                if n>1:
                    plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.0125)
                    ax=plt.subplot(221)
                    ax.set_xscale("log", nonposx='clip')
                    plt.errorbar(rres[:,0],(rres[:,1]-np.mean(rres[:,1]))*1e3,yerr=rres[:,2]*1e3,label="Mode#"+mode_name, marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)
                    plt.plot(durs, 0.0*durs, label='Mean', color='red', lw=1.5)
                    plt.xlabel('Exc. duration (µs)')
                    plt.ylabel('fr0 - '+str(round(np.mean(rres[:,1])*1e5)/100)+' (kHz)')
                    plt.legend()
                    ax=plt.subplot(223)
                    ax.set_xscale("log", nonposx='clip')
                    ax.set_yscale("log", nonposy='clip')
                    plt.errorbar(rresQ[:,0], rresQ[:,1],yerr=rresQ[:,2],label="Mode#"+mode_name, marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)
                    plt.plot(durs, 1/durs, label='Fourier limit', color='red', lw=1.5)
                    plt.xlabel('Exc. duration (µs)')
                    plt.ylabel('Res. width (MHz)')
                    plt.legend()
                    plt.show()
        self.print_footer(cal_begin)
        self.print_header('Analyse motional stability of mode #'+mode_name)
        self.fit_data(func_decay_reciprocal, dataQ, [1], plt_labels=['Exc. dur. (µs)','Res. width (MHz)','Mode #'+mode_name, 'Model: (1/texc)+bndwdth/MHz'], plt_log=[1,1]);
        return data, dataQ

    def cal_mode_freq(self, prec=1, mode=0, expperpoint=150, numofpoints=22, rng=5, **kwargs):
        cal_begin=self.print_header('Meas. mode freq. (dc tickle)')
        ret = []
        if mode==1 or mode==0:
            fr_lf,err_lf,_ ,_ ,_ ,_ = self.cal_modefr('PDQ_Tickle_BDX1', '1', 0.3/prec, 100.*prec, expperpoint=expperpoint, numofpoints=numofpoints, rng=rng, **kwargs)
            ret.extend([fr_lf,err_lf])

        if mode==2 or mode==0:    
            fr_mf,err_mf,_ ,_ ,_ ,_ = self.cal_modefr('PDQ_Tickle_BDX1', '2', 0.6/prec, 100.*prec, expperpoint=expperpoint, numofpoints=numofpoints, rng=rng, **kwargs)
            ret.extend([fr_mf,err_mf])

        if mode==3 or mode==0:
            fr_hf,err_hf,_ ,_ ,_ ,_ = self.cal_modefr('PDQ_Tickle_BDX1', '3', .35/prec, 100.*prec, expperpoint=expperpoint, numofpoints=numofpoints, rng=rng, **kwargs)
            ret.extend([fr_hf,err_hf])

        if mode==0:
            trace = round((fr_lf**2)+(fr_mf**2)+(fr_hf**2),2)
            print('Trace: %.3f' %trace)
            self.set('trace',str(trace))

        self.print_footer(cal_begin)
        return tuple(ret)
         
    def cal_mw_fr(self, script, trans_name, invert, expperpoint=250, numofpoints=17, verbose=True):
        cal_begin = self.print_header('MW trans_'+str(trans_name))
        fr=float(self.get('fr_mw_'+trans_name))    
        dur=float(self.get('t_mw_'+trans_name))
        start=fr-1.5/dur
        stop=fr+1.5/dur
        if self.count_ions()>0:
            return self.single_fit_frequency(script, 'fr_mw_'+trans_name, invert, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, verbose=verbose)
            
        else: 
            print('No ion?!')
        self.print_footer(cal_begin)
        
            
    def cal_raman_fr(self, script, trans_name, invert=True, expperpoint=250, numofpoints=15, blwp=1, **kwargs):
        fr=float(self.get('fr_'+trans_name))    
        dur=float(self.get('t_'+trans_name))
        start=fr-blwp*1.5/dur
        stop=fr+blwp*1.5/dur
        numofpoints=blwp*numofpoints    
        if self.count_ions()>0:
            return self.single_fit_frequency(script, 'fr_'+trans_name, invert, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, **kwargs)
        else: 
            print('No ion?!')
            
    def cal_raman_dur(self, script, trans_name, invert=False, expperpoint=200, numofpoints=20, fit=True):
        dur=float(self.get('t_'+trans_name))
        start=0
        stop=3.5*dur
        if self.count_ions()>0:
            if fit==True:
                return self.single_fit_time(script, 't_'+trans_name, invert, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop)
            if fit==False:
                return self.run(script, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop)
        else: 
            print('No ion?!')

    def cal_mw_dur(self,script, trans_name, invert=False, expperpoint=50, numofpoints=35, verbose=True, **kwargs):
        dur=float(self.get('t_mw_'+trans_name))
        start=0.
        stop=3.75*dur
        if self.count_ions()>0:
            return self.single_fit_time(script, 't_mw_'+trans_name, invert=invert, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, verbose=verbose, **kwargs)
        else: 
            print('No ion?!')
            
    def cal_mw_trans(self, trans_sel=None, ScanFr=0, expperpoint=250, numofpoints=15, verbose=True):
        cal_begin=self.print_header('Cal. µw transitions')
        if trans_sel==0 or trans_sel==None:
            if ScanFr==1:
                self.cal_mw_fr('0_mw_3p3_2p2_FScan', '3p3_2p2', invert=True, expperpoint=expperpoint, numofpoints=numofpoints)
                print('Freq. of 3p3 - 2p2 trans. fixed.')
            else:
                value, value_err, R2, popt, perr, chi2 = self.cal_mw_dur('1_mw_3p3_2p2_Flop', '3p3_2p2', invert=False, expperpoint=expperpoint, numofpoints=numofpoints, give_all_para=True, verbose=verbose)

                # set contrast: mw_contr_low,mw_contr_high
                A = popt[0]
                C = popt[4]
                self.set('mw_contr_high',str(round(C+A,2)));
                self.set('mw_contr_low',str(round(C-A,2)));
        if trans_sel==1 or trans_sel==None:
            if ScanFr==1:
                self.cal_mw_fr('4_mw_3p1_2p2_FScan', '3p1_2p2', invert=True, expperpoint=expperpoint, numofpoints=numofpoints, verbose=verbose)
            else:
                self.cal_mw_dur('4_mw_3p1_2p2_Flop', '3p1_2p2', invert=False, expperpoint=expperpoint, numofpoints=numofpoints, verbose=verbose)
        if trans_sel==2 or trans_sel==None:
            if ScanFr==1:
                self.cal_mw_fr('5_mw_3p1_2p0_FScan', '3p1_2p0', invert=True, expperpoint=expperpoint, numofpoints=numofpoints, verbose=verbose)
            else:    
                self.cal_mw_dur('5_mw_3p1_2p0_Flop', '3p1_2p0', invert=False, expperpoint=expperpoint, numofpoints=numofpoints, verbose=verbose)
        self.print_footer(cal_begin)
        
    def set_shim_coil_curr(self, prec=0, delta=.99, expperpoint=250, numofpoints=5, **kwargs):
        cal_begin=self.print_header('Shim quantization field (0.1 V = 2.6 mG)')
        z_cur,_=self.get_shim('ZCoilCurrent')
        if prec==0:
            if self.count_ions()>0:
                if z_cur-delta/2 < 0: 
                    start=0
                else: 
                    start=z_cur-delta/2
                curr,curr_err,gof=self.single_fit_position('0_mw_3p3_2p2_ZCoilScan', 'dummy'
                              , start=start, stop=z_cur+delta/2
                              , invert=True, expperpoint=expperpoint
                              , numofpoints=numofpoints, live_plots_flag=False
                              ,**kwargs)
                if gof>.7:
                    self.set_shim('ZCoilCurrent',str(curr))
                return curr,curr_err,gof
        if prec==1:
            if self.count_ions()>0:
                curr,curr_err,gof=self.single_fit_position('0_mw_3p3_2p2_ZCoilRamsey', 'dummy'
                              , start=z_cur-delta/2, stop=z_cur+delta/2
                              , invert=True, expperpoint=expperpoint
                              , numofpoints=numofpoints
                              ,**kwargs)
                if gof>.7:
                    self.set_shim('ZCoilCurrent',str(curr))
                return curr,curr_err,gof
        self.print_footer(cal_begin)
            
    def scan_shim(self, script, name_var, delta, **kwargs):
        pos_start,_ = self.get_shim(name_var)
        if self.count_ions()>0:
            name, data = self.run(script, start=pos_start-delta/2, stop=pos_start+delta/2, **kwargs)                
            return name, data
        
    def scan2d_shim(self, script='Z_Op_Ex_BDX', name_vars=['Ex', 'Ez'], deltas=[0.25, 0.275], nop=5, **kwargs):
        cal_begin=self.print_header('2D Shim scan of '+name_vars[0]+' and '+name_vars[1])
        posX_start,_=self.get_shim(name_vars[0])
        posY_start,_=self.get_shim(name_vars[1])
        lst=np.linspace(posY_start-deltas[1]/2,posY_start+deltas[1]/2, nop)
        ylst=[]
        zlst=[]
        n=0
        for shim in lst:    
            self.set_shim(name_vars[1], shim, verbose=False)
            name, data = self.scan_shim(script, name_vars[0], deltas[0], verbose=False, numofpoints=nop-1, rnd_sampled=False, **kwargs)
            self.set_shim(name_vars[1], posY_start, verbose=False)
            xlst=data[0]
            ylst.append(shim)
            zlst.append(data[1])
            n=n+1
            if n>1:
                X=np.array(xlst)
                Y=np.array(ylst).reshape(n)
                Z=np.array(zlst)
                clear_output(wait=True)
                self.print_header('2D Shim scan of '+name_vars[0]+' and '+name_vars[1])
                print('Run #',n,'/',nop)
                plt.pcolormesh(X, Y, Z, cmap='bwr')
                plt.xlabel(name_vars[0])
                plt.ylabel(name_vars[1])
                plt.plot(posX_start, posY_start, 'b+')
                plt.colorbar(label='Cts.')
                plt.show()
        self.print_footer(cal_begin)
        return X, Y, Z
    
    def scan2d(self, script='BDX', name_vars=['Ex', 'Ez'], deltas=[0.25, 0.275], nop=5, **kwargs):
        cal_begin=self.print_header('2D Shim scan of '+name_vars[0]+' and '+name_vars[1])
        
        posX_start,_=self.get_shim(name_vars[0])
        posY_start,_=self.get_shim(name_vars[1])
        xlst=np.linspace(posX_start-deltas[0]/2,posX_start+deltas[0]/2, nop)
        ylst=np.linspace(posY_start-deltas[1]/2,posY_start+deltas[1]/2, nop)
        
        xlst_d=[]
        ylst_d=[]
        zlst_d=[]
        n=0
        
        for shim_y in ylst:    
            for shim_x in xlst:
                xlst_d.append(shim_x)
                ylst_d.append(shim_y)
                self.set_shim(name_vars[0], shim_x, verbose=False)
                self.set_shim(name_vars[1], shim_y, verbose=False)
                name, data=self.run('BDX', verbose=False, live_plots_flag=False, expperpoint=250, numofpoints=0);
                self.set_shim(name_vars[0], posX_start, verbose=False)
                self.set_shim(name_vars[1], posY_start, verbose=False)
                #self.run('BDX', verbose=False, live_plots_flag=False, expperpoint=1, numofpoints=0);
                zlst_d.append(data[1])
                X=np.array(xlst_d)
                Y=np.array(ylst_d)
                Z=np.array(zlst_d).reshape(n+1)
                #Z=Z/np.max(Z)
                print('Pts #:', n, data[1])
                n=n+1
                clear_output(wait=True)
                self.print_header('2D Shim scan of '+name_vars[0]+' and '+name_vars[1])
                plt.scatter(X, Y, s=deltas[0]**0.5*1e3, c=Z, alpha=1, marker='o', cmap='bwr')
                plt.plot(posX_start, posY_start, 'b+')
                plt.xlabel(name_vars[0])
                plt.ylabel(name_vars[1])
                plt.colorbar(label='Cts.')
                #plt.clim(0, np.max(Z))
                #plt.gray()
                plt.show()
                
        self.print_footer(cal_begin)
        return X, Y, Z
    
    def op_shim(self, script, name_var, delta, **kwargs):
        pos_start,_ = self.get_shim(name_var)
        if self.count_ions()>0:
            pos,pos_err,gof = self.single_fit_position(script, 'dummy', invert=False, \
                                                       start=pos_start-delta/2, stop=pos_start+delta/2, \
                                                       expperpoint=250, numofpoints=6, **kwargs)
            if gof>.7:
                self.set_shim(name_var, str(pos))
                #print('Fit OK')                
            return pos,pos_err,gof
        
    def simple_comp(self, dx=0.15, dy=1.25, dz=0.25, Mesh=0.5,**kwargs):
        cal_begin= self.print_header('Shim field opt.')
        if dx!=0:
            self.op_shim('Z_Op_Ex_BDX', 'Ex', dx, **kwargs);
        if dy!=0:
            self.op_shim('Z_Op_Ey_BDX', 'Ey', dy, **kwargs);
        
        if dz!=0:
            self.op_shim('Z_Op_Ez_BDX', 'Ez', dz, **kwargs);
        if Mesh!=0:
            self.op_shim('Z_Op_Mesh_BDX', 'Mesh', Mesh, **kwargs);
        self.print_footer(cal_begin)
        
    def shim_comp_tickle(self, shimchoice='Hxz', rng=np.linspace(-0.15,0.15,7)):
        cal_begin=self.print_header('Use RF tickle to opt.: '+shimchoice)
        res=[]
        shim_cur,_=self.get_shim(shimchoice)
        rng=shim_cur+rng
        for shim in rng:
            self.print_header(shimchoice+' set to:'+str(shim))
            self.set_shim(shimchoice,str(shim))
            self.simple_comp()
            if self.check_ion()>0:
                val, err, R2, popt_1, perr_1, chi2=self.cal_modefr('PDQ_Tickle_BDX1', '1_rf', 
                                                                   .35, 75., expperpoint=100, numofpoints=22, rng=2.);
                val, err, R2, popt_2, perr_2, chi2=self.cal_modefr('PDQ_Tickle_BDX1', '2_rf', 
                                                                   .05, 10., expperpoint=100, numofpoints=33, rng=3);
                res.append([shim,popt_1[1],perr_1[1],popt_2[1],perr_2[1]])
                npres=np.array(res)
                data_1=[npres[:,0],npres[:,1],npres[:,2]]
                self.plot_data(data_1);
                plt.xlabel(shimchoice+' (a.u.)')
                plt.ylabel('Dip depth (cts.)')
                plt.show()
                data_2=[npres[:,0],npres[:,3],npres[:,4]]
                self.plot_data(data_2);
                plt.xlabel(shimchoice+' (a.u.)')
                plt.ylabel('Dip depth (cts.)')
                plt.show()
        op_shim,_,gof=self.single_fit_func(data_1, self.fit_single_gauss, invert=True, add_contrast=False, xlabel=shimchoice+' (a.u.)', ylabel='Dip depth mode #1 (cts.)');
        self.single_fit_func(data_2, self.fit_single_gauss, invert=True, add_contrast=False, xlabel=shimchoice+' (a.u.)', ylabel='Dip depth mode #2 (cts.)');
        if gof>.75:
            self.set_shim(shimchoice,str(op_shim))
        else:
            self.set_shim(shimchoice,str(shim_cur))
        self.simple_comp()
        self.print_footer(cal_begin)
        return data_1, data_2
    
    def setshim_tickle(self, shimchoice='RFMod', rng=np.linspace(-0.2,0.2,7)):
        cal_begin=self.print_header('Mode freqs as fct. of '+shimchoice)
        res=[]
        shim_cur,_=self.get_shim(shimchoice)
        rng=shim_cur+rng
        i=0
        for shim in rng:
            i+=1
            self.print_header(shimchoice+' set to:'+str(shim))
            self.set_shim(shimchoice,str(shim))
            #self.simple_comp()
            if self.check_ion()>0:
                val1, err1=self.cal_mode_freq(prec=1., mode=1, verbose=True)
                rf_pwr, rf_pwr_err=self.get_rf_pwr(500)
                val2, err2=self.cal_mode_freq(prec=1., mode=2, verbose=True)
                #rf_pwr_in, rf_pwr_in_err=self.get_rf_pwr_in(250)
                res.append([shim, val1,err1, val2,err2, rf_pwr, rf_pwr_err])
                npres=np.array(res)
                if i>1:
                    data_1=[npres[:,0],npres[:,1],npres[:,2]]
                    data_2=[npres[:,0],npres[:,3],npres[:,4]]
                    data_3=[npres[:,0],npres[:,5],npres[:,6]]
                    #data_4=[npres[:,0],npres[:,7],npres[:,8]]
                    self.single_fit_func(data_1, self.fit_linear, invert=False, add_contrast=False, xlabel=shimchoice+' (a.u.)', ylabel='Mode freq #1 (MHz)');
                    self.single_fit_func(data_2, self.fit_linear, invert=False, add_contrast=False, xlabel=shimchoice+' (a.u.)', ylabel='Mode freq #2 (MHz)');
                    self.single_fit_func(data_3, self.fit_linear, invert=False, add_contrast=False, xlabel=shimchoice+' (a.u.)', ylabel='RF pwr in (a.u.)');
                    #self.single_fit_func(data_4, self.fit_linear, invert=False, add_contrast=False, xlabel=shimchoice+' (a.u.)', ylabel='RF pwr in (a.u.)');
                

        self.set_shim(shimchoice,str(shim_cur))
        #self.simple_comp()
        self.print_footer(cal_begin)
        return data_1, data_2

    def stab_rf_pwr(self, set_val=0.071, delta=0.05, npts=5, nexp=250):
        cal_begin=self.print_header('Reference RF amp.')
        res=[]
        shimchoice='RFMod'
        shim_cur,_=self.get_shim(shimchoice)
        drng=np.linspace(-delta/2,delta/2,npts)
        np.random.shuffle(drng)
        rng=shim_cur+drng
        i=0
        for shim in rng:
            i+=1
            self.set_shim(shimchoice,str(shim), verbose=False)
            #self.simple_comp()
            if self.check_ion(verbose=False)>0:
                #clear_output(wait=True)
                rf_pwr, rf_pwr_err=self.get_rf_pwr(nexp)
                res.append([shim, (rf_pwr-set_val), rf_pwr_err/np.sqrt(nexp)])
                npres=np.array(res)
                #if i>1:
                    
                #    self.plot_data(data)
                #    plt.show()
        #clear_output(wait=True)     
        data=[npres[:,0],npres[:,1],npres[:,2]]
        value, value_err, R2, popt, perr, chi2=self.single_fit_func(data, self.fit_linear
                                         , invert=False, add_contrast=False
                                         , xlabel=shimchoice+' (a.u.)', ylabel='dP_rf (a.u.)', give_all_para=True);

        shim_op=-round(popt[1]/popt[0],4)
        
    
        if R2>0.75:
            self.set_shim(shimchoice,str(shim_op))
        else:
            self.set_shim(shimchoice,str(shim_cur))
        #self.simple_comp()
        self.print_footer(cal_begin)
        return data
        
    def get_T2(self, Ramsey_durs, script, nexp=100, npts=22):
        cal_begin = self.print_header('Ramsey T2')
        res=np.array([])
        self.set('dummy', 1.7)
        self.set_shim_coil_curr(prec=0, delta=0.75, numofpoints=5);
        self.cal_mw_trans(trans_sel=1, ScanFr=1)
        self.cal_mw_trans(trans_sel=2, ScanFr=1)
        self.cal_mw_trans(trans_sel=None, ScanFr=0)
        n=0
        for i in Ramsey_durs:
            if self.count_ions()>0:
                n=n+1
                print('------------------------------')
                print('Ramsey duration (µs): ',i)
                print('------------------------------')
                self.set('t_ramsey', str(i))
                self.set_shim_coil_curr(prec=0, delta=0.75, numofpoints=5);
                if self.count_ions()>0:
                    value, value_err, R2, popt, perr, chi2=self.single_fit_time(script, 'dummy', invert=True, expperpoint=nexp, numofpoints=npts, start=-np.pi, stop=np.pi, give_all_para=True, verbose=True)
                    res=np.append(res,(float(i), np.abs(popt[0]),perr[0]))
                    rres=res.reshape((n,3))
                if n>1:
                    plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.25)
                    ax=plt.subplot(221)
                    ax.set_xscale("log", nonposx='clip')
                    plt.errorbar(rres[:,0],rres[:,1]/rres[0,1],yerr=rres[:,2]/rres[0,1],label=script, marker = 'o', markersize=5., color='C0', lw=1.5, ls='',fmt='',capsize=3)            
                    plt.xlabel('Ramsey duration (µs)')
                    plt.ylabel('Contrast (a.u.)')
                    plt.ylim((-0.1,1.2))
                    plt.legend()
                    plt.show()
        self.print_footer(cal_begin)   
        return rres
    
    def get_Qubit_Ramsey_cnt_fringe(self, Ramsey_durs, script, nexp=100, npts=22):
        cal_begin = self.print_header('Ramsey count fringes: qubit')
        res=np.array([])
        resQ=np.array([])
        self.set('dummy', 1.7)
        n=0
        for i in Ramsey_durs:
            n=n+1
            print('------------------------------')
            print('Ramsey duration (µs): ',i)
            print('------------------------------')
            self.set_shim_coil_curr(prec=0, delta=0.45, numofpoints=5);
            self.set('t_ramsey', str(i))
            dur=1/float(i)
            cnt=float(self.get('fr_mw_3p1_2p0'))
            start=(cnt-dur/2.35)
            stop=(cnt+dur/2.35)
            value, value_err, R2, popt, perr, chi2=self.single_fit_frequency(script, 'fr_mw_3p1_2p0', invert=True, expperpoint=nexp, numofpoints=npts, start=start, stop=stop, verbose=True, give_all_para=True)
            res=np.append(res,(float(i), value, value_err))
            resQ=np.append(resQ,(float(i), popt[2], perr[2]))
            rres=res.reshape((n,3))
            rresQ=resQ.reshape((n,3))
            if n>1:
                plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.0125)
                ax=plt.subplot(221)
                ax.set_xscale("log", nonposx='clip')
                plt.errorbar(rres[:,0],(rres[:,1]-np.mean(rres[:,1]))*1e6,yerr=rres[:,2]*1e6,label=script, marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)            
                plt.xlabel('Ramsey duration (µs)')
                plt.ylabel('fr0 - '+str(round(np.mean(rres[:,1])*1e8)/100)+' (Hz)')
                plt.legend()
                ax=plt.subplot(223)
                ax.set_xscale("log", nonposx='clip')
                ax.set_yscale("log", nonposy='clip')
                ax.set_ylim(ymin=np.min(rres[:,1]/rresQ[:,1])/2,ymax=np.max(rres[:,1]/rresQ[:,1])*2)
                plt.errorbar(rres[:,0], rres[:,1]/rresQ[:,1],yerr=rres[:,1]/rresQ[:,1]*((rres[:,2]/rres[:,1])**2+(rresQ[:,2]/rresQ[:,1])**2)**0.5,label=script, marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)  
                plt.xlabel('Ramsey duration (µs)')
                plt.ylabel('Qual. factor')
                plt.legend()
                plt.show()
        self.print_footer(cal_begin)      
        return rres, rresQ
    
    def probe_qubit_freq(self, fr_3p3_2p2=1541.050):
        self.set('fr_mw_3p3_2p2',str(fr_3p3_2p2))
        self.set_shim_coil_curr(prec=0, delta=0.65, numofpoints=9);
        self.cal_mw_trans(trans_sel=1, ScanFr=1)
        self.cal_mw_trans(trans_sel=2, ScanFr=1)
        #self.cal_mw_trans(trans_sel=None, ScanFr=0)

        for t_ramsey in [100, 1000, 5000]:
            if self.count_ions()==1:
                print('-----------------')
                print('Probe dur. (µs):', t_ramsey)
                print('-----------------')
                self.set('t_ramsey', str(t_ramsey))
                dur=1/t_ramsey
                cnt=float(self.get('fr_mw_3p1_2p0'))
                start=(cnt-dur/2.35)
                stop=(cnt+dur/2.35)
                value, value_err, R2, popt, perr, chi2=self.single_fit_frequency('Qubit_Ramsey', 'fr_mw_3p1_2p0', invert=True, expperpoint=75, numofpoints=17, start=start, stop=stop, verbose=True, give_all_para=True)
        return value, value_err
               
    def find_field_indep(self, freqs):
        res=[]
        for freq in freqs:
            print('---------------------')
            print('Set freq.:', freq)
            print('---------------------')
            fr_qubit, fr_err=self.probe_qubit_freq(fr_3p3_2p2=freq)
            res.append([freq*1e3, fr_qubit*1e6, fr_err*1e6])
            self.errplt(res, lbls=['Set freq. 3p3-2p2 (kHz)','Qubit freq. (Hz)'])
        return res
    
    def get_Raman_Stark_shift(self, beam=0, man_op=False):
        cal_begin=self.print_header('Raman ac Stark shifts')
        self.set('pwr_b1',0.45)
        self.set('pwr_b2',0.45)
        self.set('pwr_r1',0.6)
        self.set('pwr_r2',0.45)
        self.set('fr_acs','1541.05')
        self.set('spin_up_mw','0')
        
#Settings for BEAM B1        
        if beam=='b1' or beam==0:
            if self.count_ions()>0:
                dur = self.get('t_b1')
                if float(dur)>300: self.set('t_b1', '300')
                dur, err, R2 = self.cal_dur('2_3p3_2p2_ACS_b1', 'b1', invert=False, expperpoint=250, numofpoints=13, verbose=True)
                if R2>.7:            
                    self.set('t_b1_pos',str(dur/2))
                    print('Op. pos at t (µs):', dur/2,'\n')
            if man_op==True and beam=='b1':
                print('--------------------------------------------------')
                print('USER ACTION NEEDED: Please optimize beam position!')
                print('--------------------------------------------------')
                self.run('2_3p3_2p2_ACS_b1_opt', verbose=False, live_plots_flag=True);
                dur = self.get('t_b1')
                if float(dur)>300: self.set('t_b1', '300')
                dur, err, R2 = self.cal_dur('2_3p3_2p2_ACS_b1', 'b1', invert=False, expperpoint=250, numofpoints=13, verbose=True)

#Settings for BEAM B2        
        if beam=='b2' or beam==0:
            if self.count_ions()>0:
                dur = self.get('t_b2')
                if float(dur)>300: self.set('t_b2', '300')
                dur, err, R2 = self.cal_dur('2_3p3_2p2_ACS_b2', 'b2', invert=False, expperpoint=250, numofpoints=13, verbose=True)
                if R2>.7:            
                    self.set('t_b2_pos',str(dur/2))
                    print('Op. pos at t (µs):', dur/2)
            if man_op==True and beam=='b2':
                print('--------------------------------------------------')
                print('USER ACTION NEEDED: Please optimize beam position!')
                print('--------------------------------------------------')
                self.run('2_3p3_2p2_ACS_b2_opt', verbose=False, live_plots_flag=True);
                dur = self.get('t_b1')
                if float(dur)>300: self.set('t_b1', '300')
                dur, err, R2 = self.cal_dur('2_3p3_2p2_ACS_b2', 'b2', invert=False, expperpoint=250, numofpoints=13, verbose=True)
                    
#Settings for BEAM R2                    
        if beam=='r2' or beam==0:
            if self.count_ions()>0:
                dur = self.get('t_r2')
                if float(dur)>300: self.set('t_r2', '300')
                dur, err, R2 = self.cal_dur('2_3p3_2p2_ACS_r2', 'r2', invert=False, expperpoint=250, numofpoints=13, verbose=True)
                if R2>.7:
                    if dur>1000:
                        op=800
                        self.set('t_r2_pos',str(op))
                    else:
                        op=dur/2
                        self.set('t_r2_pos',str(op))
                    print('Op. pos at t (µs):', op)
            if man_op==True and beam=='r2':
                print('--------------------------------------------------')
                print('USER ACTION NEEDED: Please optimize beam position!')
                print('--------------------------------------------------')
                self.run('2_3p3_2p2_ACS_r2_Opt', verbose=False, live_plots_flag=True);
                dur = self.get('t_r2')
                if float(dur)>300: self.set('t_r2', '300')
                dur, err, R2 = self.cal_dur('2_3p3_2p2_ACS_r2', 'r2', invert=False, expperpoint=250, numofpoints=13, verbose=True)
        self.print_footer(cal_begin)
    
    def sbc_settings(self, beam_conf=1, cool_m=[1,0,0], no_sbc_cycles=[12, 15, 0], t_rd=2.5):
        self.set('spin_up_cc','0')
        self.set('rsb_use_bconf', str(1))
        self.set('spin_up_mw','0')
        self.set('t_wait','0')
        self.set('beam_conf', str(beam_conf))
        #SELECT MODES FOR SBC
        self.set('cool_mode_3',str(cool_m[2]))
        self.set('cool_mode_2',str(cool_m[1]))
        self.set('cool_mode_1',str(cool_m[0]))
        #NO OF COOLING CYCLES ON SBs
        self.set('no_sbc_cycles_3',str(no_sbc_cycles[2]))
        self.set('no_sbc_cycles_2',str(no_sbc_cycles[1]))
        self.set('no_sbc_cycles_1',str(no_sbc_cycles[0]))
        #DURATION SCALINGS
        self.set('t_sbc1_faktor','0.8')
        self.set('t_sbc2_faktor','0.8')
        self.set('t_sbc3_faktor','0.8')
        #FLOP DURATION
        self.set('t_sbc_1','0')
        self.set('t_sbc_2','0')
        self.set('t_sbc_3','0')
        #RD/RP DURATION
        self.set('t_rd', str(t_rd))
        
    def cal_axial_Raman(self, beam_conf=1, cool_m=[1,0,0], no_sbc_cycles=[10, 12, 0], t_rd=2.5, precal=False):
        cal_begin=self.print_header('Cal. axial sb cooling')
        
        if precal == True:
            self.simple_comp();
            self.set_shim_coil_curr();
            self.cal_mw_trans(trans_sel=0, ScanFr=0)
            
        
        self.sbc_settings(beam_conf=beam_conf, cool_m=cool_m, no_sbc_cycles=no_sbc_cycles, t_rd=t_rd)
        self.cal_raman_fr('0_CAR_0_Scan', 'oc', expperpoint=250, numofpoints=15, blwp=1)
        #self.cal_raman_dur('0_CAR_1_Flop', 'oc')

        self.set('spin_up_mw','1')
        self.cal_raman_fr('1_RSB_1_0_Scan', '1_rsb_1', expperpoint=250, numofpoints=15, blwp=1, invert=False)
        self.cal_raman_dur('1_RSB_1_1_Flop', '1_rsb_1', invert=True)

        self.cal_raman_fr('1_RSB_2_0_Scan', '1_rsb_2', expperpoint=250, numofpoints=15, blwp=1, invert=False)
        self.cal_raman_dur('1_RSB_2_1_Flop', '1_rsb_2', invert=True)

        self.set('spin_up_mw','0')
        self.cal_raman_dur('0_CAR_1_Flop', 'oc', expperpoint=250);
        self.print_footer(cal_begin)
        
    def cal_radial_Raman(self, beam_conf=2, cool_m=[0,1,0], no_sbc_cycles=[35, 0, 0], t_rd=2.5, precal=False):
        cal_begin=self.print_header('Cal. radial sb cooling')
        self.sbc_settings(beam_conf=beam_conf, cool_m=cool_m, no_sbc_cycles=no_sbc_cycles, t_rd=t_rd)
        if precal == True:
            self.simple_comp();
            self.set_shim_coil_curr();
            self.cal_mw_trans(trans_sel=0, ScanFr=0)
            
        self.set('spin_up_mw','0')
        self.cal_raman_fr('0_CAR_0_Scan', 'oc', expperpoint=250, numofpoints=15, blwp=1)
        #self.cal_raman_dur('0_CAR_1_Flop', 'oc')

        self.set('spin_up_mw','1')
        self.cal_raman_fr('2_RSB_1_0_Scan', '2_rsb_1', expperpoint=250, numofpoints=15, blwp=1, invert=False)
        self.cal_raman_dur('2_RSB_1_1_Flop', '2_rsb_1', invert=True)

       # self.cal_raman_fr('2_RSB_2_0_Scan', '2_rsb_2', expperpoint=250, numofpoints=15, blwp=1.75, invert=False)
       # self.cal_raman_dur('2_RSB_2_1_Flop', '2_rsb_2', invert=True)

        self.set('spin_up_mw','0')
        self.cal_raman_dur('0_CAR_1_Flop', 'oc', expperpoint=250);
        self.print_footer(cal_begin)     
            
    def est_nbar_freq_scans(self, popt_b, perr_b, popt_r, perr_r):
        Cntrst=np.abs(popt_r[0])-np.abs(popt_b[0])

        Pbsb=popt_b[1]/Cntrst
        Prsb=popt_r[1]/Cntrst
        #print(Pbsb, Pbsb_err)

        Pbsb_err=Pbsb*perr_b[1]/popt_b[1]
        Prsb_err=Prsb*perr_r[1]/popt_r[1]
        #print(Prsb, Prsb_err)

        R=Prsb/Pbsb
        R_err=R*((Pbsb_err/Pbsb)**2+(Prsb_err/Prsb)**2)**0.5
        #print(R, R_err)

        nbar=round(R/(1-R),2)
        nbar_err=round(nbar*R_err/R,2)
        #print(nbar, nbar_err)
        return nbar, nbar_err

    def check_LF_temp(self, expperpoint=150, numofpoints=17, level=1, t_heat=0.0):
        cal_begin = self.print_header('Mode LF temperature')
        self.set('t_wait',str(t_heat))
        print('---------------------------------------')
        print('Mode #1 and t_heat (µs):',t_heat)
        print('---------------------------------------')
        self.set('cool_mode_1','1')
        self.set('cool_mode_2','0')
        if level==1:
            self.set('spin_up_mw','1')
            value, value_err, R2, popt_b, perr_b, chi2=self.cal_raman_fr('1_RSB_1_0_Scan', '1_rsb_1', expperpoint=expperpoint, numofpoints=numofpoints, blwp=1.25, invert=False, give_all_para=True)
            self.set('spin_up_mw','0')
            value, value_err, R2, popt_r, perr_r, chi2=self.cal_raman_fr('1_RSB_1_0_Scan', '1_rsb_1', expperpoint=expperpoint, numofpoints=numofpoints, blwp=1.25, invert=True, give_all_para=True)
            nbar, nbar_err=self.est_nbar_freq_scans(popt_b, perr_b, popt_r, perr_r)
            print('---------------------------------------')
            print('Est. thermal '+self.prnt_rslts('<n>', nbar, nbar_err))
            print('---------------------------------------')
            self.set('t_wait',str(0.0))
            self.print_footer(cal_begin)
            return nbar, nbar_err
        
        if level==2:
            self.set('spin_up_mw','1')
            self.set('A_init_spin_up','1')
            name_bsb, data_bsb = self.cal_raman_dur('1_RSB_1_1_Flop', '1_rsb_1', fit=False, expperpoint=expperpoint, numofpoints=numofpoints)
            self.plot_data(data_bsb, name_bsb)
            plt.show()
            self.set('spin_up_mw','0')
            self.set('A_init_spin_up','0')
            name_rsb, data_rsb = self.cal_raman_dur('1_RSB_1_1_Flop', '1_rsb_1', fit=False, expperpoint=expperpoint, numofpoints=numofpoints)
            self.plot_data(data_rsb, name_rsb)
            plt.show()
            print('Measurement took (h, min, sec): ', datetime.datetime.now()-cal_begin)

            print('\n Analyse data: ...')
            fr = float(self.get('fr_1'))
            return self.single_fit_sb([name_bsb, name_rsb], mode_freq=fr, mode_angle=0., Rabi_init=0.2, fock=False);
    
    def check_MF_temp(self, expperpoint=300, numofpoints=17, level=1, t_heat=0.0):
        cal_begin = self.print_header('Mode MF temperature')
        self.set('t_wait',str(t_heat))
        print('---------------------------------------')
        print('Mode #2 and t_heat (µs):',t_heat)
        print('---------------------------------------')
        self.set('cool_mode_1','0')
        self.set('cool_mode_2','1')
        if level==1:
            self.set('spin_up_mw','1')
            value, value_err, R2, popt_b, perr_b, chi2=self.cal_raman_fr('2_RSB_1_0_Scan', '2_rsb_1', expperpoint=expperpoint, numofpoints=numofpoints, blwp=1.5, invert=False, give_all_para=True)
            self.set('spin_up_mw','0')
            value, value_err, R2, popt_r, perr_r, chi2=self.cal_raman_fr('2_RSB_1_0_Scan', '2_rsb_1', expperpoint=expperpoint, numofpoints=numofpoints, blwp=1.5, invert=True, give_all_para=True)
            nbar, nbar_err=self.est_nbar_freq_scans(popt_b, perr_b, popt_r, perr_r)
            print('---------------------------------------')
            print('Est. thermal '+self.prnt_rslts('<n>', nbar, nbar_err))
            print('---------------------------------------')
            self.set('t_wait',str(0.0))
            self.print_footer(cal_begin)
            return nbar, nbar_err
        
    def heating_rate(self, durs, mode_no=1, precal=False):
        res=[]
        N=0    
        if mode_no==1:
            cal_begin=self.print_header('Take AXIAL (Mode #1) heating rate')
            print('For durations (µs):',durs)
            print('--------------------------------------- \n')
            if precal==True:
                #self.simple_comp();
                #self.cal_mode_freq(prec=0, mode=mode_no, verbose=True);
                self.set_shim_coil_curr();
                self.check_LF_temp(level=1, t_heat=0);
        if mode_no==2:
            cal_begin=self.print_header('Take RADIAL (Mode #2) heating rate')
            print('For durations (µs):',durs)
            print('--------------------------------------- \n')
            if precal==True:
                #self.simple_comp();
                #self.cal_mode_freq(prec=0, mode=mode_no, verbose=True);
                self.set_shim_coil_curr();
                self.check_MF_temp(level=1, t_heat=0);
            
        if precal==True:
            #self.simple_comp();
            #self.cal_mode_freq(prec=0, mode=mode_no, verbose=True);
            self.set_shim_coil_curr();
            self.check_LF_temp(level=1, t_heat=0);
        
        for t_heat in durs:
            if mode_no==1:
                nbar, nbar_err=self.check_LF_temp(level=1, t_heat=t_heat);
            if mode_no==2:
                nbar, nbar_err=self.check_MF_temp(level=1, t_heat=t_heat);
            if nbar>0 and nbar<3.5:
                N+=1
                res.append([t_heat, nbar, nbar_err])
            if len(res)>2:
                npres=np.array(res)
                data=[npres[:,0],npres[:,1],npres[:,2]+0.005*npres[:,1]] 
                print('\n-----------------------------------------')
                print('Heating rate measurement -', datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
                print('-----------------------------------------')
                value, value_err, R2, popt, perr, chi2=self.single_fit_func(data, self.fit_linear, 
                                                                            invert=False, 
                                                                            add_contrast=False, 
                                                                            give_all_para=True,
                                                                            xlabel='Wait duration (µs)',
                                                                            ylabel='<n>')
                print('-----------------------------------------')
                if mode_no==1:
                    print('Axial freq. (2 pi MHz):',float(self.get('fr_1')))
                    fr_oc = float(self.get('fr_oc'))
                    fr_1_rsb_1 = float(self.get('fr_1_rsb_1'))
                    fr_1_rsb_2 = float(self.get('fr_1_rsb_2'))
                    lf = np.array([fr_oc-fr_1_rsb_1,(fr_oc-fr_1_rsb_2)/2,fr_1_rsb_1-fr_1_rsb_2])
                    print('\t (via SBs):'+self.prnt_rslts('fr_1', np.mean(lf),np.std(lf)))
                if mode_no==2:
                    print('Radial freq. (2 pi MHz):',float(self.get('fr_2')))       
                print('Initial '+self.prnt_rslts('<n>', popt[1], perr[1]))
                print('Heating rate (quanta/ms) '+self.prnt_rslts('d<n>/dt', popt[0]*1000, perr[0]*1000))
                print('-----------------------------------------\n')
        if mode_no==1:
            self.set('nbar_1',str(popt[1]))
            self.set('nbar_dot_1',str(popt[0]*1000))
        if mode_no==2:
            self.set('nbar_2',str(popt[1]))
            self.set('nbar_dot_2',str(popt[0]*1000))
        self.print_footer(cal_begin)
        self.ion_report()
        return data
        
    def check_LF_temp_noise(self, n_amp, n_dur, n_shim):
        R2,popt,perr=[[],[]],[[],[]],[[],[]]
        print('---------------------------------------')
        self.create_noise_burst_wvf(noise_dur_mu=n_dur, amp=1, verbose=False);
        self.select_waveform(1, amp=n_amp);
        self.set_ips([['noise_shim',n_shim]])
        print('Noise on shim:', n_shim)
        print('---------------------------------------')
        self.sbc_settings(beam_conf=1, cool_m=[1,0,0], no_sbc_cycles=[12, 15, 0], t_rd=2.5)
        #for spin_state in [0, 1]:
        #    print('Spin state:',spin_state)
        #    _,_,R2[spin_state],popt[spin_state],perr[spin_state],_=self.run_seq(seq=['header','doppler_cool','sb_cool',
        #                                                                             'opt_mw_pulse_to_2p2','shim_noise_pulse',
        #                                                                             'raman_pulse_1_rsb_1','bdx_detect','footer'], 
        #                                                 par_type='fr', par_name='1_rsb_1', scl=1.5, nexp=100,
        #                                                 ips=[['spin_up_mw', spin_state], ['cool_mode_1',1], ['beam_conf',1]], 
        #                                                 set_op_para=False, check_script=False);
        #
        spin_state=1
        print('Spin state:',spin_state)
        val, err, R2, popt[spin_state], perr[spin_state], chi2=self.run_seq(seq=['header','doppler_cool',
                  'sb_cool',
                  'opt_mw_pulse_to_2p2','shim_noise_pulse',
                  'raman_pulse_1_rsb_1',
                  'bdx_detect','footer'], 
             par_type='fr', par_name='1_rsb_1', scl=1.5, nexp=100, 
             ips=[['spin_up_mw', spin_state], ['cool_mode_1',1], ['beam_conf',1]], 
             fit_result=True, set_op_para=False, check_script=False)

        spin_state=0
        print('Spin state:',spin_state)
        name, data=self.run_seq(seq=['header','doppler_cool',
                          'sb_cool',
                          'opt_mw_pulse_to_2p2','shim_noise_pulse',
                          'raman_pulse_1_rsb_1',
                          'bdx_detect','footer'], 
                     par_type='fr', par_name='1_rsb_1', scl=1.5, nexp=100, 
                     ips=[['spin_up_mw', spin_state], ['cool_mode_1',1], ['beam_conf',1]], 
                     fit_result=False, set_op_para=False, check_script=False)
        
        start=[-2, popt[1][3], popt[1][2], 8.5]
        limits_del=np.array(start)*np.array([-5., 1e-7, 1e-3, 2.5])
        limits = ((np.array(start)-np.array(limits_del)).tolist(),(np.array(start)+np.array(limits_del)).tolist())

        res, err,R2, chisquared=self.fit_data(sincSquare, data, 
                    start=start, limits=limits, invert=False,
                    plt_labels=['Scan para. (a.u.)','Cts. (a.u.)','data_name','model_func'], 
                    plt_log=[0,0])

        popt[spin_state], perr[spin_state]=[res[3],-res[0],res[2],res[1]], [err[3],err[0],err[2],err[1]]
        nbar, nbar_err=self.est_nbar_freq_scans(popt[1], perr[1], popt[0], perr[0])
        print('---------------------------------------')
        print('Est. thermal '+self.prnt_rslts('<n>', nbar, nbar_err))
        print('---------------------------------------')
        return nbar, nbar_err

    ## Sandbox
    def load_ions_abl(self, Try = 10):
        self.switch_to_pmt(1)
        for i in range(Try):
            self.open_photo_laser(1)
            self.pulse_abl_laser(1)
            self.open_photo_laser(0)
            cnt = self.check_ion(verbose=False)
            if cnt > 2.5:
                print('Ion!')
                break
    ## RF Opt using camera
        