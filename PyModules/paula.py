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

from PyModules.epos import EPOS
import PyModules.analyse_eios.eios_data as eios_file
from PyModules.analyse_eios.eios_data import read, read_xml, find_files, load, save
from PyModules.analyse_eios.eios_analyse import significant_digit, round_sig, plot_fit
from PyModules.analyse_eios.eios_analyse import fit_func, func_decay_exponential, func_decay_reciprocal, gauss, sincSquare, fit_linear, func_lin
from PyModules.utilities import do_async, integer_hill_climb, wait
from PyModules.pcm_stream import PCM, pcm_open, pcm_plot_hists, pcm_plot_stats
from PyModules.osci.osci import Oscilloscope

def osci_read(N_avg=10, channel=[1,2,3,4], tag='DS1Z', verbose=False):
    osci = Oscilloscope(verbose=verbose,fast=True)
    usb = osci.list_device()
    adr = usb[-1]
    for name in usb:
        if (name.find(tag)>-1):
            adr = name
            break
    if verbose:
        print('Oscilloscope Address\n\t%s'%adr)
    osci.open(adr)

    data = []
    for i in range(N_avg):
        t, data_single, dt = osci.read(channel)
        data.append(data_single.tolist())
    return t, np.array(data)

mirror_idx_b3_v = 0
mirror_idx_b3_h = 1
mirror_idx_bd_v = 2
mirror_idx_bd_h = 3
dac_idx_r2_v = 0
dac_idx_r2_h = 1
dac_idx_b1_v = 2
dac_idx_b1_h = 3

class PAULA(EPOS):

    def __init__(self, N=35, Nexpp_coarse=50, Nexpp_fine=200, counter_level=[3,9,14,20], \
                    results_path='/home/qsim/Results/tiamo4.sync', \
                    cache_path='./data/', log_file='./data/log.txt', \
                    wvf_folder='../UserData', \
                    wvf_db_file='../UserData/waveform_db.json', \
                    gof_thres=0.7, do_plot=True, do_live=False, dark=False):
        super().__init__(cache_path=cache_path,log_file=log_file, \
                            gof_thres=gof_thres, do_plot=do_plot, do_live=do_live, \
                            wvf_folder=wvf_folder, wvf_db_file=wvf_db_file)

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

    def check_ion(self, verbose=True, numofpoints=0):
        _,data=self.run('BDX', numofpoints=numofpoints, expperpoint=250, verbose=False)
        cnt = data[1][-1]
        if verbose:
            print('BDx cts:', cnt)
        return cnt

    def count_ions(self, cnts=None, verbose=False, give_all_param=False):
        if cnts is None:
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

    def switch_to_pmt(self,state, ch_pmt=5):
        self.ttl_ME(ch_pmt,state)
        time.sleep(1.)

    def close_exp_port(self,state, ch_exp=6):
        self.ttl_ME(ch_exp,state)
        time.sleep(1.)

    def close_loading_port(self,state, ch_load=7):
        self.ttl_ME(ch_load,state)
        time.sleep(1.)

    def heat_oven(self,state, ch_oven_0=2, ch_oven_1=3):
        self.ttl_ME(ch_oven_1,0)
        self.ttl_ME(ch_oven_0,state)
        time.sleep(1.)

    def load_ions(self, N, t_heat=150, timeout=180):
        cal_begin = self.print_header('Try loading %i ion(s)'%N)
        self.check_photo_freq()
        print('Switch to pmt')
        self.switch_to_pmt(1)
        self.close_exp_port(0)
        self.close_loading_port(1)
        cnt, fluo = self.count_ions(verbose=False, give_all_param=True)

        html_str = '<p style="font-family: monospace, monospace;"><b>&emsp;%s</b></p>'
        #html_str = '&emsp;<b>%s</b>'
        disp_txt = 'Counted %i ion(s) (BDX cnt %.2f)'%(cnt,fluo)
        dh = display(HTML(html_str%disp_txt),display_id=True)

        if cnt==N:
            self.print_footer(cal_begin)
            print('Already (%i) ion(s) loaded!'%N)
            return cnt==N
        try:
            self.heat_oven(1)
            print('Oven on: pre-heat')
            wait(t_heat)
            t0 = time.time()
            i = 0
            self.close_loading_port(0)
            while(True):
                i+=1
                tn = time.time()
                dt = tn-t0
                if (dt)>timeout:
                    print('Timeout (%.1f sec)'%timeout)
                    print('Check:')
                    print('\t Laser: BD (locked?)')
                    print('\t Laser: Photo (%.5f THz) (locked, blocked?)'%(525.40565))
                    print('\t PDQ state (restart?)')
                    print('\t RF signal (13dBm) & amplifier on?')
                    break
                #self.close_loading_port(0)
                data = self.run('low_RF_Exp_Load_Exp_fast', numofpoints=0, expperpoint=1, verbose=False)
                #self.close_loading_port(1)
                time.sleep(1.)
                cnt, fluo = self.count_ions(verbose=False, give_all_param=True)

                disp_txt = 'Try#%3i @ %7.2fs/Stray light %4.1f/Counted %i ion(s)--BDX %4.1f'%(i,dt,data[1][-1],cnt,fluo)
                disp_txt+='/'+self.check_photo_freq(verbose=False)
                dh.update(HTML(html_str%disp_txt))

                if cnt == N:
                    self.set('ion_loaded',str(time.time()));
                    print('Successfully loaded %i ions!'%cnt)
                    break
                elif cnt>N:
                    print('Too many ions loaded (%i)! Discard! Continue... '%cnt)
                    self.kill_ions()
            print('Oven off!?')
            self.heat_oven(0)
            print('Close loading port')
            self.close_loading_port(1)
        except:
            cnt=-1
            self.heat_oven(0)
            #raise
        self.print_footer(cal_begin)
        return cnt==N

    def update_shim(self,profile_name,shim_name,value):
        if (self.set_profile(profile_name=profile_name,shim_name=shim_name,value=str(value))):
            self.check_ion(verbose=False);
            return True
        else:
            return False

    def kill_ions(self,profile_name='EU_cool',shim_name='ch3p8'):
        shim_val,_ = self.get_profile(profile_name,shim_name)
        self.update_shim(profile_name,shim_name,0)
        self.update_shim(profile_name,shim_name,shim_val)

    def cal_dur(self, script, trans_name, invert, expperpoint=50, numofpoints=35, **kwargs):
        var_name = 't_' + trans_name
        if self.count_ions()>0:
            return super().cal_dur(script, var_name, invert, \
                           expperpoint=expperpoint, numofpoints=numofpoints, **kwargs)

    def cal_fr(self, script, trans_name, invert, expperpoint=50, numofpoints=35, **kwargs):
        var_f_name = 'fr_' + trans_name
        var_t_name = 't_' + trans_name
        if trans_name=='STR_1R':
            var_t_name = 't_lf_str_1R'
        elif trans_name=='STR_2R':
            var_t_name = 't_lf_str_2R'
        dur=float(self.get(var_t_name))
        if self.count_ions()>0:
            freq, freq_err,_ = super().cal_fr(script, var_f_name, dur, invert, \
                                     expperpoint=expperpoint, numofpoints=numofpoints, **kwargs)
            return freq, freq_err

    def cal_mode_fr(self, script, mode_name, amp, dur, expperpoint=75, numofpoints=17, **kwargs):
        var_t_name = 't_tickle_'+mode_name
        var_u_name = 'u_tickle_'+mode_name
        var_f_name = 'fr_' + mode_name

        dur_preset = self.get(var_t_name)
        self.set(var_t_name, str(dur));

        amp_preset = self.get(var_u_name)
        self.set(var_u_name, str(amp));

        if self.count_ions()>0:
            ret = super().cal_fr(script, var_f_name, dur, \
                                 invert=True, rng=4.5, \
                                 expperpoint=expperpoint, \
                                 numofpoints=numofpoints, \
                                 **kwargs)
            self.set(var_t_name,dur_preset);
            self.set(var_u_name,amp_preset);
            return ret

    def set_shim(self, shim, value, profiles=['EU_cool', 'EU_squeeze', 'EU_target']):
        for prof in profiles:
            self.set_profile(profile_name=prof,shim_name=shim,value=str(value))

    def get_shim(self, shim_name, profile_name='EU_cool'):
        value, value_date = self.get_profile(profile_name=profile_name,shim_name=shim_name)
        return value

    def opt_shim(self, script, shim, delta, update=True, give_all_para=False, **kwargs):
        E, E_err, R2, popt, perr, chi2, shim_preset, last_date = self.opt_shim_data(script, shim, delta, **kwargs)
        if (update and (R2 > self.gof_thres)):
            print('Update shim %s: %f (%f [%s])' % (shim,E,shim_preset,last_date))
            self.set_shim(shim, E)
        if give_all_para:
            return E, E_err, R2, popt, perr, chi2
        else:
            return E, E_err

    def op_bd(self, delta_v=190, delta_h=160):
        cal_begin = self.print_header('Optimize BD')
        if self.count_ions()>0:
            # pos_v-75 to pos_v+90
            self.opt_mirror('BD_pos_mirror_V', 'pos_bd_V', mirror_idx_bd_v, delta_v, numofpoints=9, expperpoint=120, invert=False, verbose=False)
        if self.count_ions()>0:
            # pos_h-60 to pos_h+85
            self.opt_mirror('BD_pos_mirror_H', 'pos_bd_H', mirror_idx_bd_h, delta_h, numofpoints=9, expperpoint=120, invert=False, verbose=False)
        if self.count_ions()>0:
            self.opt_mirror('BD_pos_mirror_V', 'pos_bd_V', mirror_idx_bd_v, delta_v, numofpoints=9, expperpoint=120, invert=False, verbose=False)
        self.check_ion();
        self.print_footer(cal_begin)

    def op_b3(self, delta_v=480, delta_h=146, ddn=3):
        cal_begin = self.print_header('Optimize B3')
        self.set('A_sbc','0')
        self.set('dynamic_decoupling_n',str(ddn))
        dur_ini = 0
        dur = 0
        if self.count_ions()>0:
            dur_ini, err, _ = self.cal_dur('B3_acs', 'b3_pos', invert=False, expperpoint=75, numofpoints=17)
            #opt_dur = dur_ini/1.5
            opt_dur = dur_ini[0]/2.
            self.set('t_b3_pos',str(opt_dur))
            print('Optimize at %f µs'%opt_dur)
        if self.count_ions()>0:
            # pos_v-240 to pos_v+240
            self.opt_mirror('B3_pos_mirror_V', 'pos_b3_V', mirror_idx_b3_v, delta_v, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            # pos_h-60 to pos_h+85
            self.opt_mirror('B3_pos_mirror_H', 'pos_b3_H', mirror_idx_b3_h, delta_h, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            self.opt_mirror('B3_pos_mirror_V', 'pos_b3_V', mirror_idx_b3_v, delta_v, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            dur, err, _ = self.cal_dur('B3_acs', 'b3_pos', invert=False, expperpoint=75, numofpoints=17)
            self.set('t_b3_pos',str(dur[0]/1.5))
        self.check_ion();
        print('Improved by approx. (%):', round(100*(dur_ini[0]-dur[0])/dur_ini[0],3))
        self.set('dynamic_decoupling_n',str(1))
        self.print_footer(cal_begin)

    def op_b1(self, delta_v=2, delta_h=2,ddn=2):
        #ddn: number of Pi-cycles
        cal_begin = self.print_header('Optimize B1')
        self.set('A_sbc','0')
        self.set('dynamic_decoupling_n',str(ddn))
        dur_ini = 0
        dur = 0
        if self.count_ions()>0:
            dur_ini, err, _ = self.cal_dur('B1_acs', 'b1_pos', invert=False, expperpoint=75, numofpoints=17)
            opt_dur = dur_ini[0]/2
            self.set('t_b1_pos',str(opt_dur))
            print('Optimize at %f µs'%opt_dur)
        if self.count_ions()>0:
            self.opt_dac('B1_pos_mirror_V', 'pos_b1_V', dac_idx_b1_v, delta_v, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            self.opt_dac('B1_pos_mirror_H', 'pos_b1_H', dac_idx_b1_h, delta_h, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            self.opt_dac('B1_pos_mirror_V', 'pos_b1_V', dac_idx_b1_v, delta_v, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            dur, err, _ = self.cal_dur('B1_acs', 'b1_pos', invert=False, expperpoint=75, numofpoints=17)
            self.set('t_b1_pos',str(dur[0]/1.5))
        self.check_ion();
        print('Improved by approx. (%):', round(100*(dur_ini[0]-dur[0])/dur_ini[0],3))
        self.set('dynamic_decoupling_n',str(1))
        self.print_footer(cal_begin)

    def op_r2(self, delta_v=2, delta_h=2,ddn=1, A_sbc=0):
        #ddn: number of Pi-cycles
        cal_begin = self.print_header('Optimize R2')
        self.set('A_sbc',str(A_sbc))
        self.set('dynamic_decoupling_n',str(ddn))
        dur_ini = 0
        dur = 0
        if self.count_ions()>0:
            dur_ini, err, _ = self.cal_dur('R2_acs', 'r2_pos', invert=False, expperpoint=75, numofpoints=17)
            opt_dur = dur_ini[0]/1.5
            self.set('t_r2_pos',str(opt_dur))
            print('Optimize at %f µs'%opt_dur)
        if self.count_ions()>0:
            self.opt_dac('R2_pos_mirror_V', 'pos_r2_V', dac_idx_r2_v, delta_v, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            self.opt_dac('R2_pos_mirror_H', 'pos_r2_H', dac_idx_r2_h, delta_h, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            self.opt_dac('R2_pos_mirror_V', 'pos_r2_V', dac_idx_r2_v, delta_v, numofpoints=11, expperpoint=75, invert=True, verbose=False)
        if self.count_ions()>0:
            dur, err, _ = self.cal_dur('R2_acs', 'r2_pos', invert=False, expperpoint=75, numofpoints=17)
            self.set('t_r2_pos',str(dur[0]/1.5))
        self.check_ion();
        print('Improved by approx. (%):', round(100*(dur_ini[0]-dur[0])/dur_ini[0],3))
        self.set('dynamic_decoupling_n',str(1))
        self.print_footer(cal_begin)

    def op_rf_freq(self):
        from IPython.display import clear_output, display
        x=[]
        y=[]
        yerr=[]
        i=0
        for i in range(100):
            clear_output(wait=True)
            x.append(i)
            ty, tyerr = self.read_avg_adc_me(ch=3, no_avg=200, verbose=False)
            y.append(ty)
            yerr.append(tyerr)
            print(round(ty,3))
            i+=1
            time.sleep(.3)

    def print_laser_level(self):
        print('--------------------------')
        print('UV laser pwr level (a.u.):')
        print('--------------------------')
        # BD: 28
        p_bd = self.read_adc_me(28)
        r_bd = max(0.,p_bd/0.05)*100
        print('BD port #4: %.0f%%'%(r_bd))
        #RD: ch 30
        p_rd = self.read_adc_me(30)
        r_rd = max(0.,p_rd/0.0155)*100
        print('RD: %.0f%%'%(r_rd))
        # Photo: 29
        p_ph = self.read_adc_me(29)
        r_ph = max(0.,p_ph/0.5)*100
        print('Photo: %.0f%%'%(r_ph))
        print('--------------------------')

    def print_drive_rf_level(self):
        print('--------------------------')
        print('Trap drive pwr level (a.u.):')
        print('--------------------------')
        print('Main line refl.: ', max(0.,round(self.read_adc_me(3)/1.8, 3)))
        print('Pick up (oven): ', max(0.,round(self.read_adc_me(4), 3)))
        print('--------------------------')

    def print_shim_settings(self, profile_name='EU_cool'):
        def print_shim_list(shim_list,profile_name,unit=''):
            n = np.max([len(s) for s in shim_list])
            msg = r'%s '+'(%s): '%(unit)+r'%10f [%s]'
            for shim_name in shim_list:
                value, value_date = self.get_profile(profile_name=profile_name,shim_name=shim_name)
                print(msg%(shim_name.ljust(n), value, value_date))
        print('--------------------------')
        print('Profile: ', profile_name)
        print('--------------------------')
        ch3p8, ch3p8_date = self.get_profile(profile_name=profile_name,shim_name='ch3p8')
        print('RF pwr / ch3p8 (a.u.): %.3f [%s]'%(ch3p8, ch3p8_date))
        print('--------------------------')
        print_shim_list(['ERaO', 'EZO', 'ERiO','ERaU', 'EZU', 'ERiU'],profile_name,r'V')
        print_shim_list(['ExExpZ', 'EyExpZ', 'EzExpZ'],profile_name,r'kV/m')
        print_shim_list(['FinestHzz'],profile_name,'MHz^2')
        #print('--------------------------')
        #print_shim_list(['ExExpZ_fine', 'EyExpZ_fine', 'EzExpZ_fine'],profile_name,r'V/m')
        #print_shim_list(['HxyExpZ_fine', 'HxzExpZ_fine', 'HzzExpZ_fine'],profile_name,'MHz^2')
        print('')

    def set_comp_sett(self, ExExpZ=0, EyExpZ=0, EzExpZ=0,
                            Hxy_comp_07_2018=0, Hxz_comp_07_2018=0,
                            FinestHzz=0, Hxy_at=0,
                            ExExpZ_fine=0, EyExpZ_fine=0,
                            Hxz_03_2018=0, Hxz_comp_07_2018_fine_EZ=0, Hxz_comp_07_2018_fine_ERU_EZ=0,
                            ch3p8=3.141, profile_name='EU_cool'):
        self.set_shim('ch3p8', str(ch3p8))

        self.set_shim('ExExpZ', str(ExExpZ))
        self.set_shim('EyExpZ', str(EyExpZ))
        self.set_shim('EzExpZ', str(EzExpZ))

        self.set_shim('FinestHzz', str(FinestHzz))

        self.set_shim('ExExpZ_fine', str(ExExpZ_fine))
        self.set_shim('EyExpZ_fine', str(EyExpZ_fine))
        self.set_shim('EzExpZ_fine', str(EzExpZ_fine))
        self.set_shim('HxyExpZ_fine', str(HxyExpZ_fine))
        self.set_shim('HxzExpZ_fine', str(HxzExpZ_fine))
        self.set_shim('HzzExpZ_fine', str(HzzExpZ_fine))
        self.print_shim_settings(profile_name)

    def print_mw_prop(self):
        fr_mw_3p3_2p2, fr_mw_3p3_2p2_date = self.get_parameter('fr_mw_3p3_2p2')
        t_coherence_3p3_2p2, t_coherence_3p3_2p2_date = self.get_parameter('t_coherence_3p3_2p2')
        fr_mw_3p1_2p2, fr_mw_3p1_2p2_date = self.get_parameter('fr_mw_3p1_2p2')
        fr_mw_3p1_2p0, fr_mw_3p1_2p0_date = self.get_parameter('fr_mw_3p1_2p0')
        fr_mw_3p0_2p0, fr_mw_3p0_2p0_date = self.get_parameter('fr_mw_3p0_2p0')
        t_coherence_3p0_2p0, t_coherence_3p0_2p0_date = self.get_parameter('t_coherence_3p0_2p0')
        print('--------------------------')
        print('Hyperfine transitions (2pi MHz)')
        print('--------------------------')
        print('3p3 - 2p2: %.3f [%s]'%(fr_mw_3p3_2p2,fr_mw_3p3_2p2_date))
        print('3p1 - 2p2: %.3f [%s]'%(fr_mw_3p1_2p2,fr_mw_3p1_2p2_date))
        print('3p1 - 2p0: %.3f [%s]'%(fr_mw_3p1_2p0,fr_mw_3p1_2p0_date))
        print('3p0 - 2p0: %.3f [%s]'%(fr_mw_3p0_2p0,fr_mw_3p0_2p0_date))
        print('--------------------------')
        print('Coherence times (µs)')
        print('--------------------------')
        print('3p3 - 2p2: %.3f [%s]'%(t_coherence_3p3_2p2,t_coherence_3p3_2p2_date))
        print('3p3 - 2p2: %.3f [%s]'%(t_coherence_3p0_2p0,t_coherence_3p0_2p0_date))
        print('')

    def print_raman_prop(self):
        t_b1, t_b1_date = self.get_parameter('t_b1')
        t_b3, t_b3_date = self.get_parameter('t_b3')
        t_r2, t_r2_date = self.get_parameter('t_r2')

        print('--------------------------')
        print('Raman AC Stark shifts (kHz)')
        print('--------------------------')
        print('B1: %.1f [%s]'%(500/t_b1, t_b1_date))
        print('B3: %.1f [%s]'%(500/t_b3, t_b3_date))
        print('R2: %.1f [%s]'%(500/t_r2, t_r2_date))
        print('')

    def print_mode_prop(self, Nion=2):
        fr_hf, fr_hf_date = self.get_parameter('fr_hf')
        fr_mf, fr_mf_date = self.get_parameter('fr_mf')
        fr_lf, fr_lf_date = self.get_parameter('fr_lf')
        n_hf, n_hf_date = self.get_parameter('n_th_hf')
        n_mf, n_mf_date = self.get_parameter('n_th_mf')
        n_lf, n_lf_date = self.get_parameter('n_th_lf')
        rabi_lf, rabi_lf_date = self.get_parameter('rabi_lf')
        rabi_mf, rabi_mf_date = self.get_parameter('rabi_mf')
        rabi_hf, rabi_hf_date = self.get_parameter('rabi_hf')
        if Nion==1:
            print('--------------------------')
            print('Single-Ion Freq. (2pi MHz)')
            print('--------------------------')
            print('HF COM: %.4f [%s]'%(fr_hf,fr_hf_date))
            print('MF COM: %.4f [%s]'%(fr_mf,fr_mf_date))
            print('LF    : %.4f [%s]'%(fr_lf,fr_lf_date))
            print('--------------------------')
            print('SBC to <n> (quanta)')
            print('--------------------------')
            print('HF: %.2f [%s]'%(n_hf,n_hf_date))
            print('MF: %.2f [%s]'%(n_mf,n_mf_date))
            print('LF: %.2f [%s]'%(n_lf,n_lf_date))
            print('--------------------------')
            print('Rabi rate (kHz)')
            print('--------------------------')
            print('HF: %.3f [%s]'%(rabi_hf,rabi_hf_date))
            print('MF: %.3f [%s]'%(rabi_mf,rabi_mf_date))
            print('LF: %.3f [%s]'%(rabi_lf,rabi_lf_date))
            print('')
            return fr_hf, fr_mf, fr_lf, n_hf, n_mf, n_lf

        if Nion==2:
            fr_hf_rck=(fr_hf**2-fr_lf**2)**0.5
            fr_mf_rck=(fr_mf**2-fr_lf**2)**0.5
            fr_str=3**0.5*fr_lf
            print('--------------------------------')
            print('Two-Ion Crystal Freq. (2pi MHz)')
            print('--------------------------------')
            print('HF COM: %.3f [%s]'%(fr_hf,fr_hf_date))
            print('HF RCK: %.3f' % (fr_hf_rck))
            print('MF COM: %.3f [%s]'%(fr_mf,fr_mf_date))
            print('MF RCK: %.3f' % (fr_mf_rck))
            print('STR   : %.3f' % (fr_str))
            print('LF    : %.3f [%s]'%(fr_lf,fr_lf_date))
            print('')
            return fr_hf, fr_hf_rck, fr_mf, fr_mf_rck, fr_str, fr_lf

    def ion_report(self, profile_name='EU_cool', Nion=1):
        self.print_header('Ion Report');
        timestamp=float(self.get('ion_loaded'))
        timestr = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print('Ion(s) loaded:', timestr)
        self.print_shim_settings(profile_name)
        self.print_mode_prop(Nion)
        self.print_raman_prop()
        self.print_mw_prop()

    def check_repump(self, quick=True, verbose=False):
        if verbose:
            cal_begin = self.print_header('Check Repumper');

        N_ion = self.count_ions()

        t_rd = float(self.get('t_rd_sbc'))
        t_rp = t_rd
        if quick:
            numofpoints=0
            expperpoint=75
            start_rd=t_rd-0.1
            stop_rd=t_rd+0.1
            start_rp=t_rp-0.1
            stop_rp=t_rp+0.1
        else:
            numofpoints=9
            expperpoint=50
            start_rd=0
            stop_rd=t_rd*1.5
            start_rp=0
            stop_rp=t_rp*1.5

        #self.set('A_sbc','0')

        _,data=self.run('RD_pump', numofpoints=numofpoints, expperpoint=expperpoint, start=start_rd, stop=stop_rd, live_plots_flag=False, verbose=False)
        cnts_rd = data[1][-1]
        rd = (self.count_ions(cnts=cnts_rd) == N_ion)
        if not quick:
            self.plot_data(data);
            plt.show()

        _,data=self.run('RP_pump', numofpoints=numofpoints, expperpoint=expperpoint, start=start_rp, stop=stop_rp, live_plots_flag=False, verbose=False)
        cnts_rp = data[1][-1]
        rp = (self.count_ions(cnts=cnts_rp) == 0)
        if not quick:
            self.plot_data(data);
            plt.show()

        if verbose:
            test_str = ['failed' , 'OK']
            print('RD: %s (%.2f)'%(test_str[rd],cnts_rd))
            print('RP: %s (%.2f)'%(test_str[rp],cnts_rp))
            self.print_footer(cal_begin)

        return [rd, rp]

    def get_Raman_Stark_shift(self, t_b1=180,t_b3=150,t_r2=25, SBC=0):
        cal_begin = self.print_header('Raman AC Stark shift')

        self.std_SBC()
        self.set('A_sbc',str(SBC))
        self.set('t_b1', str(t_b1))
        self.set('t_b3', str(t_b3))
        self.set('t_r2', str(t_r2))

        self.set('pwr_b1','0.6')
        self.set('pwr_b1_sbc','0.6')
        if self.count_ions()>0:
            dur, err, R2 = self.cal_dur('B1_acs', 'b1', invert=False, expperpoint=50, numofpoints=11)
            if R2>.7:
                self.set('t_b1_pos',str(dur/2))
        self.set('pwr_b3','0.6')
        self.set('pwr_b3_sbc','0.6')
        if self.count_ions()>0:
            dur, err, R2 = self.cal_dur('B3_acs', 'b3', invert=False, expperpoint=50, numofpoints=11)
            if R2>.7:
                self.set('t_b3_pos',str(dur/2))
        self.set('pwr_r2','0.5')
        self.set('pwr_r2_sbc','0.5')
        if self.count_ions()>0:
            dur, err, R2 = self.cal_dur('R2_acs', 'r2', invert=False, expperpoint=50, numofpoints=15)
            if R2>.7:
                self.set('t_r2_pos',str(dur/2))
        self.print_footer(cal_begin)

    def do_long_term(self, Nlong = 15, twait=30):
        # take initial values...
        t0 = time.time()
        if self.count_ions()>0:
            #Ex,Ex_err, Ey,Ey_err, Ez,Ez_err=self.check_ion_pos(0.635,0.05,.05,.25,numofpoints=11, expperpoint=100, itera=1);
            Ex,Ex_err, Ey,Ey_err, Ez,Ez_err=0.,0., 0.,0., 0.,0.
            fr_lf,lf_err,_=self.cal_mode_fr('PDQ_LF_FScan', 'lf', 1., 100);
            fr_mf,mf_err,_=self.cal_mode_fr('PDQ_MF_FScan', 'mf', .4, 400);
            fr_hf,hf_err,_=self.cal_mode_fr('PDQ_HF_FScan', 'hf', .27, 400);
            resa = np.array([0.,Ex,Ex_err, Ey,Ey_err, Ez,Ez_err,fr_lf,lf_err,fr_mf,mf_err,fr_hf,hf_err])

            # long term loop...
            for i in range(Nlong):
                wait(twait)
                dt=(time.time()-t0)/60
                print('Run#:',i, 'at (min):',dt)
                if self.count_ions()>0:
                    #Ex,Ex_err, Ey,Ey_err, Ez,Ez_err=self.check_ion_pos(0.635,0.05,.05,.25,numofpoints=11, expperpoint=100, itera=1);
                    Ex,Ex_err, Ey,Ey_err, Ez,Ez_err=0.,0., 0.,0., 0.,0.
                    fr_lf,lf_err,_=self.cal_mode_fr('PDQ_LF_FScan', 'lf', 1., 100);
                    fr_mf,mf_err,_=self.cal_mode_fr('PDQ_MF_FScan', 'mf', .4, 400);
                    fr_hf,hf_err,_=self.cal_mode_fr('PDQ_HF_FScan', 'hf', .27, 400);
                resa=np.append(resa,[[dt, Ex,Ex_err, Ey,Ey_err, Ez,Ez_err,fr_lf,lf_err,fr_mf,mf_err,fr_hf,hf_err]])
                res=resa.reshape((i+2,13))
                plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.025)
                ax=plt.subplot(221)
                plt.errorbar(res[:,0],(res[:,1]-res[0,1])*1000,yerr=res[:,2]*1000,label='Ex+'+str(resa[1]*1000)+'V/m', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
                plt.errorbar(res[:,0],(res[:,3]-res[0,3])*1000,yerr=res[:,4]*1000,label='Ey+'+str(resa[3]*1000)+'V/m', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
                plt.errorbar(res[:,0],(res[:,5]-res[0,5])*1000,yerr=res[:,6]*1000,label='Ez+'+str(resa[5]*1000)+'V/m', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
                plt.ylabel('Variation of stray fields (V/m) ')
                plt.xlabel('Long-term duration (min) ')
                plt.ylim(-10.5,10.5)
                plt.legend()
                plt.grid()
                ax=plt.subplot(223)
                plt.errorbar(res[:,0],(res[:,7]-res[0,7])*1000,yerr=res[:,8]*1000,label='lf+'+str(round(resa[7]*1000,4))+'kHz', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
                plt.errorbar(res[:,0],(res[:,9]-res[0,9])*1000,yerr=res[:,10]*1000,label='mf+'+str(round(resa[9]*1000,4))+'kHz', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
                plt.errorbar(res[:,0],(res[:,11]-res[0,11])*1000,yerr=res[:,12]*1000,label='hf+'+str(round(resa[11]*1000,4))+'kHz', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
                plt.ylabel('Variation of mode freq. (2pi kHz)')
                plt.xlabel('Long-term duration (min) ')
                #plt.ylim(-6.5,5)
                plt.legend()
                plt.grid()
                plt.show()
            return resa, res
        else:
            return print('Ion?')

    def shim_scan_range(self, cnt, delta, steps):
        rng = np.array([])
        stps=np.arange(cnt-delta/steps,cnt-delta/2,-2*delta/steps)
        rng = np.append(rng,[stps])
        stps = np.arange(cnt-delta/2,cnt+delta/2+delta/steps,2*delta/steps)
        rng = np.append(rng,stps)
        stps = np.arange(cnt+delta/2-delta/steps,cnt,-2*delta/steps)
        rng = np.append(rng,stps)
        rng = np.append(rng,[cnt])
        return rng

    def get_mode_fr_vs_shim(self, shim_name, cnt, delta, steps):
        cal_begin = self.print_header('Mode freq. vs shim')
        scan_range = self.shim_scan_range(cnt, delta, steps)
        self.set('A_sbc','0')
        self.set('A_init_spin_up','0')
        self.set('EU_beam_conf','0')
        self.update_shim(profile_name='EU_cool', shim_name=shim_name, value=cnt)
        self.set_shim(shim_name,cnt)
        fr,err,_=self.cal_mode_fr('LowRF_LF_FScan', 'lf', 2., 30);
        resa0=np.array([cnt,fr,err])
        fr,err,_=self.cal_mode_fr('LowRF_MF_FScan', 'mf', 4.45, 30, numofpoints=17, expperpoint=100);
        resa1=np.array([cnt,fr,err])
        fr,err,_=self.cal_mode_fr('LowRF_HF_FScan', 'hf', 5.75, 30, numofpoints=17, expperpoint=100);
        resa2=np.array([cnt,fr,err])
        npts=1
        self.set('A_sbc','0')
        for i in scan_range:
            print('----------------------------------')
            print('scan par (',shim_name,'): ',str(i))
            print('----------------------------------\n')
            #self.set_shim(shim_name,i)
            self.update_shim(profile_name='EU_cool', shim_name=shim_name, value=i)
            fr,err,_=self.cal_mode_fr('LowRF_LF_FScan', 'lf', 2., 30);
            resa0=np.append(resa0,[i,fr,err])
            fr,err,_=self.cal_mode_fr('LowRF_MF_FScan', 'mf', 4.45, 30, numofpoints=17, expperpoint=100);
            resa1=np.append(resa1,[i,fr,err])
            fr,err,_=self.cal_mode_fr('LowRF_HF_FScan', 'hf', 5.75, 30, numofpoints=17, expperpoint=100);
            resa2=np.append(resa2,[i,fr,err])
            npts=npts+1
            print(resa2)
            res0=resa0.reshape((npts,3))
            res1=resa1.reshape((npts,3))
            res2=resa2.reshape((npts,3))

            plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.025)
            ax=plt.subplot(221)
            plt.errorbar(res0[:,0],(res0[:,1]),yerr=res0[:,2],label='lf mode', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res1[:,0],(res1[:,1]),yerr=res1[:,2],label='mf mode', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res2[:,0],(res2[:,1]),yerr=res2[:,2],label='hf mode', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
        #    plt.errorbar(res0[:,0],res0[:,1],yerr=res0[:,2],label='Dip depth', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
        #    plt.errorbar(res1[:,0],res1[:,1],yerr=res1[:,2],label='Width', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
        #    plt.errorbar(res2[:,0],res2[:,1],yerr=res2[:,2],label='Centre', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
            plt.ylabel('Fit res.')
            plt.legend()
            plt.grid()
            ax=plt.subplot(223)
            #plt.errorbar(res0[:,0],res0[:,1],yerr=res0[:,2],label='Dip depth', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res0[:,0],(res0[:,1]-res0[0,1])*1000,yerr=res0[:,2]*1000,label='lf mode', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res1[:,0],(res1[:,1]-res1[0,1])*1000,yerr=res1[:,2]*1000,label='mf mode', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res2[:,0],(res2[:,1]-res2[0,1])*1000,yerr=res2[:,2]*1000,label='hf mode', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
            plt.ylabel('Variation of res (a.u.)')
            plt.xlabel(shim_name+'(a.u.)')
            plt.legend()
            plt.grid()
            plt.show()
        self.print_footer(cal_begin)
        return res0, res1, res2

    def get_mode_fr_vs_rfpwr(self, shim_name, cnt, delta, steps):
        cal_begin=self.print_header('Mode freq. vs rf power')
        scan_range = self.shim_scan_range(cnt, delta, steps)
        self.set('A_sbc','0')
        self.set('A_init_spin_up','0')
        self.set('EU_beam_conf','0')
        self.update_shim(profile_name='EU_squeeze', shim_name=shim_name, value=cnt)
        fr,err,_=self.cal_mode_fr('LowRF_LF_FScan', 'lf', .8, 100);
        resa0=np.array([cnt,fr,err])
        fr,err,_=self.cal_mode_fr('LowRF_MF_FScan', 'mf', 4.45, 30, numofpoints=17, expperpoint=100);
        resa1=np.array([cnt,fr,err])
        fr,err,_=self.cal_mode_fr('LowRF_HF_FScan', 'hf', 5.75, 30, numofpoints=17, expperpoint=100);
        resa2=np.array([cnt,fr,err])
        npts=1
        self.set('A_sbc','0')
        for i in scan_range:
            print('----------------------------------')
            print('scan par (',shim_name,'): ',str(i))
            print('----------------------------------\n')
            #self.set_shim(shim_name,i)
            self.update_shim(profile_name='EU_squeeze', shim_name=shim_name, value=i)
            fr,err,_=self.cal_mode_fr('LowRF_LF_FScan', 'lf', 0.8, 100);
            resa0=np.append(resa0,[i,fr,err])
            fr,err,_=self.cal_mode_fr('LowRF_MF_FScan', 'mf', 4.45, 30, numofpoints=17, expperpoint=100);
            resa1=np.append(resa1,[i,fr,err])
            fr,err,_=self.cal_mode_fr('LowRF_HF_FScan', 'hf', 5.75, 30, numofpoints=17, expperpoint=100);
            resa2=np.append(resa2,[i,fr,err])
            npts=npts+1
            res0=resa0.reshape((npts,3))
            res1=resa1.reshape((npts,3))
            res2=resa2.reshape((npts,3))

            plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.025)
            ax=plt.subplot(221)
            plt.errorbar(res0[:,0],(res0[:,1]),yerr=res0[:,2],label='lf mode', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res1[:,0],(res1[:,1]),yerr=res1[:,2],label='mf mode', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res2[:,0],(res2[:,1]),yerr=res2[:,2],label='hf mode', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
            plt.ylabel('Fit res.')
            plt.legend()
            plt.grid()
            ax=plt.subplot(223)
            #plt.errorbar(res0[:,0],res0[:,1],yerr=res0[:,2],label='Dip depth', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res0[:,0],(res0[:,1]-res0[0,1])*1000,yerr=res0[:,2]*1000,label='lf+'+str(round(res0[0,1]*1000,3))+' kHz', marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res1[:,0],(res1[:,1]-res1[0,1])*1000,yerr=res1[:,2]*1000,label='mf+'+str(round(res1[0,1]*1000,3))+' kHz', marker = 'o', markersize=5., color='C1', lw=1.5, ls='--',fmt='',capsize=3)
            plt.errorbar(res2[:,0],(res2[:,1]-res2[0,1])*1000,yerr=res2[:,2]*1000,label='hf+'+str(round(res2[0,1]*1000,3))+' kHz', marker = 'o', markersize=5., color='C2', lw=1.5, ls='--',fmt='',capsize=3)
            plt.ylabel('Variation of res (a.u.)')
            plt.xlabel(shim_name+'(a.u.)')
            plt.legend()
            plt.grid()
            plt.show()

        self.print_footer(cal_begin)
        return res0, res1, res2

    def check_ion_pos(self, amp, deltaX, deltaY, deltaZ, wf=4, numofpoints=9, expperpoint=100, verbose=False, itera=1, update=True, give_all_para=False, A_init_spin_up=0, A_sbc=0, EU_beam_conf=0):
        cal_begin = self.print_header('Check ion position: update? %s'%update)
        print('___Set comp. fields___')

        self.select_waveform(wf, amp)
        self.set('A_init_spin_up',str(A_init_spin_up));
        self.set('A_sbc',str(A_sbc));
        self.set('EU_beam_conf',str(EU_beam_conf));
        if EU_beam_conf>19:
            self.set('cool_calc_Q','0');
            self.set('cool_mode_lf','0');
            self.set('cool_RCKlf','0');
            self.set('cool_mode_mf','1');
            self.set('cool_mode_hf','1');

        Ex,Ex_err, Ey,Ey_err, Ez,Ez_err = None,None, None,None, None,None
        R2_X, popt_X, perr_X, chi2_X, R2_Y, popt_Y, perr_Y, chi2_Y, R2_Z, popt_Z, perr_Z, chi2_Z= None,None, None,None,None,None, None,None,None,None, None,None
        for i in range(itera):
            if self.count_ions()>0 and deltaX !=0:
                Ex, Ex_err, R2_X, popt_X, perr_X, chi2_X = self.opt_shim('EU_Ex', 'ExExpZ', deltaX, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose, update=update, give_all_para=True)
            if self.count_ions()>0 and deltaY !=0:
                Ey, Ey_err, R2_Y, popt_Y, perr_Y, chi2_Y = self.opt_shim('EU_Ey', 'EyExpZ', deltaY, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose, update=update, give_all_para=True)
            if self.count_ions()>0 and deltaZ !=0:
                Ez, Ez_err, R2_Z, popt_Z, perr_Z, chi2_Z = self.opt_shim('EU_Ez', 'EzExpZ', deltaZ, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose, update=update, give_all_para=True)
        self.check_ion();
        self.print_footer(cal_begin)
        if give_all_para:
            return R2_X, popt_X, perr_X, chi2_X, R2_Y, popt_Y, perr_Y, chi2_Y, R2_Z, popt_Z, perr_Z, chi2_Z
        else:
            return Ex,Ex_err, Ey,Ey_err, Ez,Ez_err

    def check_ion_pos_fine(self, amp, deltaX, deltaY, deltaZ, wf=4, numofpoints=9, expperpoint=100, verbose=False, itera=1, update=True, give_all_para=False, A_init_spin_up=0, A_sbc=0, EU_beam_conf=0):
        cal_begin = self.print_header('Check ion position: update? %s'%update)
        print('___Set comp. fields___')

        self.select_waveform(wf, amp)
        self.set('A_init_spin_up',str(A_init_spin_up));
        self.set('A_sbc',str(A_sbc));
        self.set('EU_beam_conf',str(EU_beam_conf));
        if EU_beam_conf>19:
            self.set('cool_calc_Q','0');
            self.set('cool_mode_lf','0');
            self.set('cool_RCKlf','0');
            self.set('cool_mode_mf','1');
            self.set('cool_mode_hf','1');

        Ex,Ex_err, Ey,Ey_err, Ez,Ez_err = None,None, None,None, None,None
        R2_X, popt_X, perr_X, chi2_X, R2_Y, popt_Y, perr_Y, chi2_Y, R2_Z, popt_Z, perr_Z, chi2_Z= None,None, None,None,None,None, None,None,None,None, None,None
        for i in range(itera):
            if self.count_ions()>0 and deltaX !=0:
                Ex, Ex_err, R2_X, popt_X, perr_X, chi2_X = self.opt_shim('EU_Ex_fine', 'ExExpZ_fine', deltaX, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose, update=update, give_all_para=True)
            if self.count_ions()>0 and deltaY !=0:
                Ey, Ey_err, R2_Y, popt_Y, perr_Y, chi2_Y = self.opt_shim('EU_Ey_fine', 'EyExpZ_fine', deltaY, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose, update=update, give_all_para=True)
            if self.count_ions()>0 and deltaZ !=0:
                Ez, Ez_err, R2_Z, popt_Z, perr_Z, chi2_Z = self.opt_shim('EU_Ez_fine', 'EzExpZ_fine', deltaZ, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose, update=update, give_all_para=True)
        self.check_ion();
        self.print_footer(cal_begin)
        if give_all_para:
            return R2_X, popt_X, perr_X, chi2_X, R2_Y, popt_Y, perr_Y, chi2_Y, R2_Z, popt_Z, perr_Z, chi2_Z
        else:
            return Ex,Ex_err, Ey,Ey_err, Ez,Ez_err

    def check_ion_curv(self, amp, deltaX, deltaY, wf=12, numofpoints=9, expperpoint=100, verbose=False, itera=1):
        cal_begin = self.print_header('Check ion curvature')
        print('___Set curv.___')

        self.select_waveform(wf, amp)
        self.set('A_init_spin_up','0');
        self.set('A_sbc','0');
        self.set('EU_beam_conf','0');
        Hxy,Hxy_err, Hxz,Hxz_err = None,None, None,None
        for i in range(itera):
            if self.count_ions()>0:
                Hxy,Hxy_err=self.opt_shim('EU_Hxy', 'Hxy_comp_07_2018', deltaX, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose)
            if self.count_ions()>0:
                Hxz,Hxz_err=self.opt_shim('EU_Hxz', 'Hxz_comp_07_2018', deltaY, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose)
        self.check_ion();
        self.print_footer(cal_begin)
        return Hxy,Hxy_err, Hxz,Hxz_err

    def check_radial_ion_pos_quantum(self,amp, deltaX, deltaY, deltaXY, beamconf, SF=1, wf=4, numofpoints=9, expperpoint=100, verbose=True):
        cal_begin = self.print_header('Radial comp fields (using ROC)')

        self.select_waveform(wf, amp)
        self.set('A_init_spin_up',str(SF));
        self.set('A_sbc','1');
        self.std_SBC()
        self.set('EU_beam_conf',str(beamconf));

        if self.count_ions()>0:
            Ex,Ex_err=self.opt_shim('EU_Ex', 'ExExpZ', deltaX, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose)
        if self.count_ions()>0:
            Ey,Ey_err=self.opt_shim('EU_Ey', 'EyExpZ', deltaY, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose)
        if self.count_ions()>0:
            Hxz,Hxz_err=self.opt_shim('EU_Hxz', 'Hxz_comp_07_2018', deltaXY, numofpoints=numofpoints, expperpoint=expperpoint, verbose=verbose)
        self.check_ion();
        self.print_footer(cal_begin)
        return Ex,Ex_err, Ey,Ey_err

    def cal_mw_trans(self, comp=0, verbose=False):
        cal_begin = self.print_header('MW transitions %i'%comp)

        self.set('A_sbc','0');
        self.set('A_init_spin_up','0');

        print('___Get 3p3 -> 2p2 MW transition___ ')
        self.cal_fr('0_3p3_2p2_FScan', 'mw_3p3_2p2', invert=True, expperpoint=60, numofpoints=11, verbose=verbose);
        value, value_err, R2, popt, perr, chi2 = self.cal_dur('0_3p3_2p2_Flop', 'mw_3p3_2p2', invert=False, \
                                                                expperpoint=60, numofpoints=19, \
                                                                give_all_para=True, verbose=verbose);

        # set contrast: mw_contr_low,mw_contr_high
        A = popt[0]
        C = popt[4]
        self.set('mw_contr_high',str(C+A));
        self.set('mw_contr_low',str(C-A));

        print('___Get 2p2 -> 3p1 MW transition___ ')
        self.cal_fr('1_3p1_2p2_FScan', 'mw_3p1_2p2', invert=False, expperpoint=60, numofpoints=11, verbose=verbose);
        self.cal_dur('1_3p1_2p2_Flop', 'mw_3p1_2p2', invert=True, expperpoint=60, numofpoints=17, verbose=verbose);
        if comp == 1:
            print('___Get 2p0 -> 3p1 MW transition___ ')
            self.cal_fr('2_3p1_2p0_FScan', 'mw_3p1_2p0', invert=True, expperpoint=60, numofpoints=9, verbose=verbose);
            self.cal_dur('2_3p1_2p0_Flop', 'mw_3p1_2p0', invert=False, expperpoint=60, numofpoints=15, verbose=verbose);
            print('___Get 2p0 -> 3p0 MW transition___ ')
            print(self.set('clk_shelf','0'))
            self.cal_fr('3_3p0_2p0_FScan', 'mw_3p0_2p0', invert=False, expperpoint=60, numofpoints=9, verbose=verbose);
            self.cal_dur('3_3p0_2p0_Flop', 'mw_3p0_2p0', invert=True, expperpoint=60, numofpoints=15, verbose=verbose);
            print('___Get 2m1 -> 3p0 MW transition___ ')
            self.cal_fr('4_3p0_2pm1_FScan', 'mw_3p0_2pm1', invert=True, expperpoint=60, numofpoints=9, verbose=verbose);
            self.cal_dur('4_3p0_2pm1_Flop', 'mw_3p0_2pm1', invert=False, expperpoint=60, numofpoints=15, verbose=verbose);

        self.print_footer(cal_begin)

    def cal_mode_freq(self, prec=1, Level=0):
        cal_begin = self.print_header('Mode frequency')

        self.set('A_sbc','0');
        self.set('A_init_spin_up','0');
        ret = []
        if Level==1 or Level==0:
            # LF
            self.set('tickle_shim_lf','ERaO');
            fr_lf,fr_lf_err,_=self.cal_mode_fr('PDQ_LF_FScan', 'lf', 1./prec, 100*prec);
            ret.extend([fr_lf,fr_lf_err])

        if Level==2 or Level==0:
            # MF
            self.set('tickle_shim_mf','EZO');
            fr_mf,fr_mf_err,_=self.cal_mode_fr('PDQ_MF_FScan', 'mf', 1./prec, 100*prec);
            ret.extend([fr_mf,fr_mf_err])

        if Level==3 or Level==0:
            # HF
            self.set('tickle_shim_hf','EZO');
            fr_hf,fr_hf_err,_=self.cal_mode_fr('PDQ_HF_FScan', 'hf', 1.25/prec, 100*prec);
            ret.extend([fr_hf,fr_hf_err])

        self.print_footer(cal_begin)
        return tuple(ret)


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
        cal_begin = self.print_header('Mode coherence')
        res=np.array([])
        self.cal_mode_fr('PDQ_LF_FScan', 'lf', 1./prec, 100*prec);
        self.cal_mode_fr('PDQ_MF_FScan', 'mf', 1.8/prec, 100*prec);
        self.cal_mode_fr('PDQ_HF_FScan', 'hf', 1.2/prec, 100*prec);

        n=0
        for i in Ramsey_durs:
            if self.count_ions()>0:
                print('------------------------------')
                print('Ramsey duration (µs): ',i)
                print('------------------------------')
                #self.cal_mode_fr('PDQ_Tickle_BDX1', mode_name, amp, dur, expperpoint=expperpoint, numofpoints=14, rng=3)
                popt, perr, R2 = self.tickle_ramsey('PDQ_Tickle_phasescan', mode_name, amp, dur, i, \
                                                    expperpoint=expperpoint, numofpoints=numofpoints, \
                                                    rnd_sampled=rnd_sampled)
                if R2>.005:
                    n=n+1
                    res=np.append(res,(float(i), np.abs(popt[0]),perr[0]))
                    rres=res.reshape((n,3))
                    data=[rres[:,0], rres[:,1]/np.max(rres[:,1]), rres[:,2]/np.max(rres[:,1])]
                if n>1:
                    self.plot_data(data)
                    plt.xlabel('Ramsey dur. (µs)')
                    plt.ylabel('Contrast (a.u.)')
                    plt.show()

        self.print_footer(cal_begin)
        self.print_header('Analyse mode #'+mode_name+' coherence')
        try:
            lbl = ['Ramsey dur. (µs)','Contrast (a.u.)','Mode #'+mode_name, 'Model: Decay w. exp']
            popt, perr, R2, chi2 = self.fit_data(func_decay_exponential, data, start, plt_labels=lbl, plot_residuals=True);
        except:
            popt = np.array(start)
            perr = 0.0*popt
            R2, chi2 = 0., 0.
        self.print_footer(cal_begin)
        return data, popt, perr, R2, chi2

    def get_mode_stab(self, mode_name, durs, expperpoint=250, numofpoints=11, rnd_sampled=False):
        cal_begin = self.print_header('Mode stability')
        res=np.array([])
        n=0
        res=np.array([])
        resQ=np.array([])
        #durs= sorted(durs, key = lambda x: random.random())
        for i in durs:
            if self.count_ions()>0:
                print('------------------------------')
                print('Exc. duration (µs): ',i)
                print('------------------------------')
                prec=i/100
                if mode_name == '1':
                    self.set('tickle_shim_lf','ERaO');
                    value, value_err, R2, popt, perr, chi2=self.cal_mode_fr('PDQ_LF_FScan', 'lf', 0.75/prec, 100*prec, expperpoint=250, numofpoints=11, give_all_para=True);
                if mode_name == '2':
                    self.set('tickle_shim_mf','EZO');
                    value, value_err, R2, popt, perr, chi2=self.cal_mode_fr('PDQ_MF_FScan', 'mf', 1.2/prec, 100*prec, expperpoint=250, numofpoints=11, give_all_para=True);
                if mode_name == '3':
                    self.set('tickle_shim_hf','EZO');
                    value, value_err, R2, popt, perr, chi2=self.cal_mode_fr('PDQ_HF_FScan', 'hf', 0.8/prec, 100*prec, expperpoint=250, numofpoints=11, give_all_para=True);
                if R2>0.5:
                    n=n+1
                    res=np.append(res,(float(i), value[0], value_err[0]))
                    resQ=np.append(resQ,(float(i), popt[2], perr[2]))
                    rres=res.reshape((n,3))
                    rresQ=resQ.reshape((n,3))
                    data=[rres[:,0], rres[:,1], rres[:,2]]
                    dataQ=[rresQ[:,0], rresQ[:,1], rresQ[:,2]]
                #self.print_header('Check mode stability')
                if n>1:
                    plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.0125)
                    ax=plt.subplot(221)
                    ax.set_xscale("log", nonposx='clip')
                    plt.errorbar(rres[:,0],(rres[:,1]-np.mean(rres[:,1]))*1e3, yerr=rres[:,2]*1e3,label="Mode#"+mode_name, marker = 'o', markersize=7.5, color='navy', lw=1.5, ls='',fmt='',capsize=0.)
                    plt.plot(durs, 0.0*durs, label='Mean', color='Orange', lw=3., alpha=0.3)
                    plt.xlabel('Exc. duration (µs)')
                    plt.ylabel('fr0 - '+str(round(np.mean(rres[:,1])*1e5)/100)+' (kHz)')
                    plt.legend()
                    ax=plt.subplot(223)
                    ax.set_xscale("log", nonposx='clip')
                    ax.set_yscale("log", nonposy='clip')
                    plt.errorbar(rresQ[:,0], rresQ[:,1],yerr=rresQ[:,2],label="Mode#"+mode_name, marker = 'o', markersize=5., color='navy', lw=1.5, ls='',fmt='',capsize=3)
                    plt.plot(durs, 1/durs, label='Fourier limit', color='Orange', lw=3., alpha=0.3)
                    plt.xlabel('Exc. duration (µs)')
                    plt.ylabel('Res. width (MHz)')
                    plt.legend()
                    plt.show()
        self.print_footer(cal_begin)

        self.print_header('Analyse motional stability of mode #'+mode_name)
        self.fit_data(func_decay_reciprocal, dataQ, [1], plt_labels=['Exc. dur. (µs)','Res. width (MHz)','Mode #'+mode_name, 'Model: (1/texc)+bndwdth/MHz'], plt_log=[1,1]);
        self.print_footer(cal_begin)
        return data, dataQ

    def cal_rad_SBC_single_ion(self, EstFreqs=False, Level=None):
        cal_begin = self.print_header('Radial SBC single ion')
        if EstFreqs==True:
            print('___Get radial carrier freq.___')
            self.set('A_sbc','0');
            self.set('A_init_spin_up','1');
            fr_roc,_=self.cal_fr('Rad0_Car_FScan', 'roc', invert=False, expperpoint=50, numofpoints=11);

            #print('___Get mode freq.___')
            #self.set('A_init_spin_up','0');
            #fr_mf,_,_=self.cal_mode_fr('PDQ_MF_FScan', 'mf', 4.2, 50);
            #fr_hf,_,_=self.cal_mode_fr('PDQ_HF_FScan', 'hf', 2.1, 50);

            print('___Set SB freqs. single ion___')
            fr_mf_1R=fr_roc - float(self.get('fr_mf'))
            self.set_parameter("fr_mf_1R", fr_mf_1R)
            fr_mf_2R=fr_roc - 2*float(self.get('fr_mf'))
            self.set_parameter("fr_mf_2R", fr_mf_2R)

            fr_hf_1R=fr_roc - float(self.get('fr_hf'))
            self.set_parameter("fr_hf_1R", fr_hf_1R)
            fr_hf_2R=fr_roc - 2*float(self.get('fr_hf'))
            self.set_parameter("fr_hf_2R", fr_hf_2R)

        if self.check_repump()==[1,1]:
            print('___Check SBC___')
            self.std_SBC(Nions=1, mode=2)
            self.set('A_init_spin_up','1');
            if Level==1 or Level==None:
                fr_mf_1R,_=self.cal_fr('Rad1_1R_MF_FScan', 'mf_1R', invert=False, expperpoint=40, numofpoints=11);
                dur, _, _ = self.cal_dur('Rad1_1R_MF_Flop', 'mf_1R', invert=True, expperpoint=40, numofpoints=15);
                self.set('t_sbc_mf_1',str(0.75*dur[0]))
                #self.set('EU_t_mf_1R',str(0.45*dur))
                self.set('EU_t_mf_1R',str(1.00*dur[0]))

            if Level==2 or Level==None:
                fr_hf_1R,_=self.cal_fr('Rad1_1R_HF_FScan', 'hf_1R', invert=False, expperpoint=40, numofpoints=11);
                dur, _, _ = self.cal_dur('Rad1_1R_HF_Flop', 'hf_1R', invert=True, expperpoint=40, numofpoints=15);
                self.set('t_sbc_hf_1',str(0.75*dur[0]))
                #self.set('EU_t_hf_1R',str(0.45*dur))
                self.set('EU_t_hf_1R',str(1.00*dur[0]))

            if Level==0 or Level==None:
                fr_roc,_=self.cal_fr('Rad0_Car_FScan', 'roc', invert=False, expperpoint=40, numofpoints=11);
                dur, _, _ = self.cal_dur('Rad0_Car_Flop', 'roc', invert=True, expperpoint=40, numofpoints=17);
                self.set('EU_t_roc',str(dur[0]))
        else:
            print('Check ion and/or RD/RP')
        self.print_footer(cal_begin)

    def cal_ax_SBC_two_ions(self, EstFreqs=False):
        cal_begin = self.print_header('Axial SBC two ion')
        if EstFreqs==True:
            print('___Get radial carrier freq.___')
            self.set('A_sbc','0');
            self.set('A_init_spin_up','1');
            fr_oc,_=self.cal_fr('0_Car_FScan', 'oc', invert=False, expperpoint=40, numofpoints=11);

            fr_lf=float(self.get('fr_lf'))
            print('___Set SB freqs. two ions___')
            fr_lf_1R=fr_oc - fr_lf
            fr_STR_1R=fr_oc - 3**0.5*fr_lf
            self.set_parameter("fr_lf_1R", fr_lf_1R)
            self.set_parameter("fr_STR_1R", fr_STR_1R)
            fr_lf_2R=fr_oc - 2.0025*fr_lf
            fr_STR_2R=fr_oc - 2.0028*3**0.5*fr_lf
            self.set_parameter("fr_lf_2R", fr_lf_2R)
            self.set_parameter("fr_STR_2R", fr_STR_2R)

        if self.check_repump()==[1,1]:
            print('___Check SBC___')
            self.set('A_sbc','1');
            self.set('A_init_spin_up','1');
            self.set('cool_lf_calc_Q','0');
            self.set('cool_mode_lf','1');
            self.set('cool_RCKlf','1');
            self.set('cool_mode_mf','0');
            self.set('cool_mode_hf','0');

            #1st SBs
            fr_lf_1R,_=self.cal_fr('1_1R_FScan', 'lf_1R', invert=False, expperpoint=40, numofpoints=11);
            dur, _, _ = self.cal_dur('1_1R_Flop', 'lf_1R', invert=True, expperpoint=40, numofpoints=20);
            self.set('t_sbc_lf_1',str(0.75*dur))
            #Stretch
            fr_STR_1R,_=self.cal_fr('6_1R_STR_FScan', 'STR_1R', invert=False, expperpoint=40, numofpoints=11);
            dur, _, _ = self.cal_dur('6_1R_STR_Flop', 'lf_str_1R', invert=True, expperpoint=40, numofpoints=20);
            self.set('t_sbc_STR_1',str(0.75*dur))

            #2nd SBs
            fr_lf_2R,_=self.cal_fr('2_2R_FScan', 'lf_2R', invert=False, expperpoint=40, numofpoints=11);
            dur, _, _ = self.cal_dur('2_2R_Flop', 'lf_2R', invert=True, expperpoint=40, numofpoints=20);
            self.set('t_sbc_lf_2',str(0.75*dur))
            #Stretch
            #fr_STR_2R,_=self.cal_fr('6_2R_STR_FScan', 'STR_2R', False, expperpoint=40, numofpoints=11);
            #dur, _, _ = self.cal_dur('6_2R_STR_Flop', 'lf_str_2R', invert=True, expperpoint=40, numofpoints=20);
            fr_str=float(self.get('fr_STR_1R'))
            fr_oc=float(self.get('fr_oc'))
            self.set('fr_STR_2R',str(fr_oc+2*(fr_oc-fr_str)))
            self.set('lf_str_2R',str(dur))
            self.set('t_sbc_STR_2',str(0.75*dur))

            #Car
            fr_oc,_=self.cal_fr('0_Car_FScan', 'oc', invert=False, expperpoint=40, numofpoints=11);
            dur, _, _ = self.cal_dur('0_Car_Flop', 'oc', invert=True, expperpoint=40, numofpoints=20);
        else:
            print('Check ion and/or RD/RP')
        self.print_footer(cal_begin)

    def cal_ax_SBC_single_ion(self, EstFreqs=False, Level=None):
        cal_begin = self.print_header('Axial SBC single ion')
        if EstFreqs==True:
            print('___Get radial carrier freq.___')
            self.set('A_sbc','0');
            self.set('A_init_spin_up','1');
            fr_oc,_=self.cal_fr('0_Car_FScan', 'oc', invert=False, expperpoint=40, numofpoints=11);
            fr_oc = fr_oc[0]
            #print('___Get mode freq.___')
            #self.set('A_init_spin_up','0');
            #fr_lf,_,_=self.cal_mode_fr('PDQ_LF_FScan', 'lf', 1., 200);
            fr_lf=float(self.get('fr_lf'))

            print('___Set SB freqs. single ion___')
            fr_lf_1R=fr_oc - fr_lf
            self.set_parameter("fr_lf_1R", fr_lf_1R)
            fr_lf_2R=fr_oc - 2.002*fr_lf
            self.set_parameter("fr_lf_2R", fr_lf_2R)

        print('___Check SBC___')
        self.std_SBC(Nions=1, mode=1)
        self.set('A_init_spin_up','1');

        if self.check_repump()==[1,1]:
            if Level==1 or Level==None:
                #1st SB
                fr_lf_1R,_=self.cal_fr('1_1R_FScan', 'lf_1R', invert=False, expperpoint=40, numofpoints=11);
                dur, _, _ = self.cal_dur('1_1R_Flop', 'lf_1R', invert=True, expperpoint=40, numofpoints=20);
                self.set('t_sbc_lf_1',str(0.9*dur[0]))

            if Level==2 or Level==None:
                #2nd SB
                fr_lf_2R,_=self.cal_fr('2_2R_FScan', 'lf_2R', invert=False, expperpoint=40, numofpoints=11);
                dur, _, _ = self.cal_dur('2_2R_Flop', 'lf_2R', invert=True, expperpoint=40, numofpoints=20);
                self.set('t_sbc_lf_2',str(0.9*dur[0]))

            if Level==0 or Level==None:
                #Car
                fr_oc,_=self.cal_fr('0_Car_FScan', 'oc', invert=False, expperpoint=40, numofpoints=11);
                dur, _, _ = self.cal_dur('0_Car_Flop', 'oc', invert=True, expperpoint=40, numofpoints=20);

        else:
            print('Check ion and/or RD/RP')
        self.print_footer(cal_begin)

    def mode_analyse(self, script, t_mode, f_mode, angle_mode, rabi, Npis, Npts, Nexp, fock=False, fit_async=False, fitit=1):
        self.set('A_sbc','1');
        self.set('A_init_spin_up','0');
        start=0.
        stop=Npis*2*t_mode
        if self.check_repump()==[1,1]:
            print('RD/RP OK')
            name,data = self.run(script, numofpoints=Npis*Npts, expperpoint=Nexp, start=start, stop=stop, counter=None)
            self.plot_data(data, name)
            print(name)
            plt.show()
            if fitit==1:
                if fit_async:
                    do_async(self.single_fit_sb, [name], mode_freq=f_mode, mode_angle=angle_mode, Rabi_init=rabi, fock=fock);
                    return None
                else:
                    return self.single_fit_sb([name], mode_freq=f_mode, mode_angle=angle_mode, Rabi_init=rabi, fock=fock)
            else:
                return 0.0, 0.0, 0.0, 0.0

    def mode_analyse_axial_lf(self, Npis, Npts, Nexp, fock=False, fit_async=False, angle=0, fitit=1):
        self.std_SBC(Nions=1, mode=1)
        t_lf=float(self.get('t_lf_1R'))
        f_lf = float(self.get('fr_lf'))
        rabi = 1.5/float(self.get('t_lf_1R'))
        return self.mode_analyse('1_1R_LF_MA', t_lf, f_lf, angle, rabi, Npis, Npts, Nexp, fock=fock, fit_async=fit_async, fitit=fitit)

    def mode_analyse_radial_mf(self, Npis, Npts, Nexp, fock=False, fit_async=False, angle=45, fitit=1):
        self.std_SBC(Nions=1, mode=2)
        t_mf=float(self.get('t_mf_1R'))
        f_mf = float(self.get('fr_mf'))
        rabi = 2./t_mf
        return self.mode_analyse('Rad1_1R_MF_MA', t_mf, f_mf, angle, rabi, Npis, Npts, Nexp, fock=fock, fit_async=fit_async, fitit=fitit)

    def mode_analyse_radial_hf(self, Npis, Npts, Nexp, fock=False, fit_async=False, angle=45, fitit=1):
        self.std_SBC(Nions=1, mode=2)
        t_hf=float(self.get('t_hf_1R'))
        f_hf = float(self.get('fr_hf'))
        rabi = 2./t_hf
        return self.mode_analyse('Rad1_1R_HF_MA', t_hf, f_hf, angle, rabi, Npis, Npts, Nexp, fock=fock, fit_async=fit_async, fitit=fitit)

    def check_LF_temp(self,Npis, Npts, Nexp, fock=False, angle=0, fitit=1):
        cal_begin = self.print_header('Mode LF temperature')
        #self.mode_analyse_axial(Npis, Npts, Nexp, fock=fock, fit_async=True);
        self.do_live=True
        red_chi, fit_valid, value, error = self.mode_analyse_axial_lf(Npis, Npts, Nexp, fock=fock, fit_async=False, angle=angle, fitit=fitit);
        if fit_valid:
            rabi_lf = round(value[0],2)
            n_th_lf = round(value[4],2)
            self.set_parameter("rabi_lf", rabi_lf)
            self.set_parameter("n_th_lf", n_th_lf)
        self.do_live=False
        self.print_footer(cal_begin)

    def check_MF_temp(self,Npis, Npts, Nexp, fock=False, angle=45, fitit=1):
        cal_begin = self.print_header('Mode MF temperature')
        self.do_live=True
        #self.mode_analyse_radial_mf(Npis, Npts, Nexp, fock=fock, fit_async=True);
        red_chi, fit_valid, value, error = self.mode_analyse_radial_mf(Npis, Npts, Nexp, fock=fock, fit_async=False, angle=angle, fitit=fitit);
        if fit_valid:
            rabi_mf = round(value[0],2)
            n_th_mf = round(value[4],2)
            self.set_parameter("rabi_mf", rabi_mf)
            self.set_parameter("n_th_mf", n_th_mf)
        self.do_live=False
        self.print_footer(cal_begin)

    def check_HF_temp(self,Npis, Npts, Nexp, fock=False, angle=45, fitit=1):
        cal_begin = self.print_header('Mode HF temperature')
        #self.mode_analyse_radial_hf(Npis, Npts, Nexp, fock=fock, fit_async=True);
        self.do_live=True
        red_chi, fit_valid, value, error = self.mode_analyse_radial_hf(Npis, Npts, Nexp, fock=fock, fit_async=False, angle=angle, fitit=fitit);
        if fit_valid:
            rabi_hf = round(value[0],2)
            n_th_hf = round(value[4],2)
            self.set_parameter("rabi_hf", rabi_hf)
            self.set_parameter("n_th_hf", n_th_hf)
        self.do_live=False
        self.print_footer(cal_begin)

    def ModRF_squeeze_MF(self,beamconf, exc_dur, exc_amp, det, numofpoints=11, expperpoint=75):
        cal_begin = self.print_header('Modulate RF: squeeze MF mode')
        fr_mf=float(self.get('fr_mf'))
        print('MF freq: ', fr_mf)
        start=fr_mf-.75/exc_dur+det
        stop=fr_mf+.75/exc_dur+det
        self.set('A_sbc','1');
        self.set('cool_mode_lf','0');
        self.set('cool_mode_mf','1');
        self.set('cool_mode_hf','1');
        self.set('A_init_spin_up','0');
        self.set('EU_beam_conf',str(beamconf));
        self.set('t_tickle_mf_EU',str(exc_dur));
        self.set('u_tickle_mf_EU',str(exc_amp));
        t_mf_1R_pre = float(self.get('t_mf_1R'))
        self.set('EU_t_mf_1R',str(0.65*t_mf_1R_pre))
        ret = [0,0,0]
        if self.check_repump()==[1,1]:
            best_value,err,gof=self.single_fit_frequency('EURFMod_MF_1R', 'fr_mf_sq', par_name='fr_mf_sq', invert=True, numofpoints=numofpoints, expperpoint=expperpoint, start=start, stop=stop, live_plots_flag=False)
            self.set('EU_t_mf_1R',str(t_mf_1R_pre))
            print('Detuning due to modulation (kHz): ',round(1e3*(best_value-fr_mf),3))
            ret = [best_value,err,gof]
        else:
            print('Check ion and/or RD/RP')
        self.print_footer(cal_begin)
        return ret

    def ModRF_squeeze_MF_ana(self,beamconf, exc_dur, exc_amp, exc_freq, numofpoints=4*7, expperpoint=75):
        cal_begin = self.print_header('Modulate RF: squeeze MF mode: analyse')
        dur=float(self.get('t_mf_1R'))
        start=0
        stop=6.*dur
        self.set('A_sbc','1');
        self.set('cool_mode_lf','0');
        self.set('cool_mode_mf','1');
        self.set('cool_mode_hf','1');
        self.set('EU_beam_conf',str(beamconf));
        self.set('t_tickle_mf_EU',str(exc_dur));
        self.set('u_tickle_mf_EU',str(exc_amp));
        self.set('fr_mf_sq',str(exc_freq))
        ret = None
        if self.check_repump()==[1,1]:
            name, data=self.run('EURFMod_MF_1R_opt', numofpoints=numofpoints, expperpoint=expperpoint, start=start, stop=stop, counter=None, live_plots_flag=True)
            print(name)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();
            ret = name
        else:
            print('Check ion and/or RD/RP')
        self.print_footer(cal_begin)
        return ret

    '''
    def opt_mirror_script(self,pos,idx,script,numofpoints=5,expperpoint=250):
        for i,p in zip(idx,pos):
            old_pos = self.get_mirror_pos(i)
            new_pos = int(p)
            #print(new_pos, old_pos, new_pos-old_pos, ((new_pos-old_pos) != 0))
            if (new_pos-old_pos) != 0:
                #print('Mirror %i: %i (%i)'%(i, new_pos, old_pos))
                self.mirror_pos(i,new_pos)
        _,data=self.run(script, numofpoints=numofpoints, expperpoint=expperpoint, verbose=False)
        x,y,err=data
        y=np.mean(y[1:])
        #print('mean count: %.3f'%y)
        return y

    def optimize_BD(self, x0=[0,0], limits=[(-200,200),(-200,200)], maxfev=200, stag_tol=6):
        idx=[2,3]
        func = lambda pos: self.opt_mirror_script(pos=pos,idx=idx,script='BD',numofpoints=1,expperpoint=2*self.Nexpp_fine)
        cal_begin = self.print_header('Optimize BD: integer_hill_climb')
        pos, cnt, _, _ = integer_hill_climb(func, x0=x0, inc=5, limits=limits, maxfev=maxfev, to=stag_tol)
        print('Set final pos: ', pos, cnt)
        for i,p in zip(idx,pos):
            self.mirror_pos(i,p)
        self.print_footer(cal_begin)
        return pos,cnt

    def optimize_B3(self, x0=[0,0], limits=[(-200,200),(-200,200)], maxfev=200, stag_tol=4):
        idx=[0,1]
        dur, err, _ = self.cal_dur(script='B3_acs', trans_name='b3_pos', invert=False)
        opt_dur = dur/2.
        self.set('t_b3_pos',str(opt_dur))
        print('Optimize at %f µs'%opt_dur)
        func = lambda pos: -self.opt_mirror_script(pos=pos,idx=idx,script='B3_pos',numofpoints=2,expperpoint=self.Nexpp_coarse)
        cal_begin = self.print_header('Optimize B3: integer_hill_climb')
        pos, cnt, _, _ = integer_hill_climb(func, x0=x0, inc=20, limits=limits, maxfev=200, to=6)
        print('Set final pos: ', pos, cnt)
        for i,p in zip(idx,pos):
            self.mirror_pos(i,p)
        self.print_footer(cal_begin)
        return pos,cnt
    '''

    def get_T2(self, Ramsey_durs, script, nexp=100, npts=22):
        cal_begin = self.print_header('Ramsey T2')
        res=np.array([])
        self.set('dummy', 1.7)
        n=0
        for i in Ramsey_durs:
            n=n+1
            print('------------------------------')
            print('Ramsey duration (µs): ',i)
            print('------------------------------')
            self.set('t_ramsey', i)
            value, value_err, R2, popt, perr, chi2=self.single_fit_time(script, 'dummy', invert=False, expperpoint=nexp, numofpoints=npts, start=0, stop=4.*np.pi, give_all_para=True, verbose=False)
            res=np.append(res,(float(i), np.abs(popt[0]),perr[0]))
            rres=res.reshape((n,3))
            if n>1:
                plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.25)
                ax=plt.subplot(221)
                plt.errorbar(rres[:,0],rres[:,1]/rres[0,1],yerr=rres[:,2]/rres[0,1],label=script, marker = 'o', markersize=5., color='C0', lw=1.5, ls='--',fmt='',capsize=3)
                plt.xlabel('Ramsey duration (µs)')
                plt.ylabel('Contrast (a.u.)')
                plt.ylim((-0.1,1.2))
                plt.legend()
                plt.show()
        self.print_footer(cal_begin)
        return rres

    def get_Ramsey_cnt_fringe(self, Ramsey_durs, script, nexp=100, npts=22):
        cal_begin = self.print_header('Ramsey count fringes')
        res=np.array([])
        resQ=np.array([])
        self.set('dummy', 1.7)
        n=0
        for i in Ramsey_durs:
            n=n+1
            print('------------------------------')
            print('Ramsey duration (µs): ',i)
            print('------------------------------')
            self.set('t_ramsey', i)
            dur=1/float(i)
            cnt=float(self.get('fr_mw_3p0_2p0'))
            start=(cnt-dur/2.25)
            stop=(cnt+dur/2.25)
            value, value_err, R2, popt, perr, chi2=self.single_fit_frequency(script, 'fr_mw_3p0_2p0', invert=False, expperpoint=400, numofpoints=15, start=start, stop=stop, verbose=False, give_all_para=True)
            res=np.append(res,(float(i), value, value_err))
            resQ=np.append(resQ,(float(i), popt[2], perr[2]))
            rres=res.reshape((n,3))
            rresQ=resQ.reshape((n,3))
            if n>1:
                plt.subplots_adjust(bottom=0.0, right=2, top=1.75, hspace=0.0125)
                ax=plt.subplot(221)
                ax.set_xscale("log", nonposx='clip')
                plt.errorbar(rres[:,0],(rres[:,1]-np.mean(rres[:,1]))*1e6,yerr=rres[:,2]*1e6,label=script, marker = 'o', markersize=5., color='navy', lw=1.5, ls='--',fmt='',capsize=3)
                plt.xlabel('Ramsey duration (µs)')
                plt.ylabel('fr0 - '+str(round(np.mean(rres[:,1])*1e8)/100)+' (Hz)')
                plt.legend()
                ax=plt.subplot(223)
                ax.set_xscale("log", nonposx='clip')
                ax.set_yscale("log", nonposy='clip')
                ax.set_ylim(ymin=np.min(rres[:,1]/rresQ[:,1])/2,ymax=np.max(rres[:,1]/rresQ[:,1])*2)
                plt.errorbar(rres[:,0], rres[:,1]/rresQ[:,1],yerr=rres[:,1]/rresQ[:,1]*((rres[:,2]/rres[:,1])**2+(rresQ[:,2]/rresQ[:,1])**2)**0.5,label=script, marker = 'o', markersize=5., color='navy', lw=1.5, ls='--',fmt='',capsize=3)
                plt.xlabel('Ramsey duration (µs)')
                plt.ylabel('Qual. factor')
                plt.legend()
                plt.show()
        self.print_footer(cal_begin)
        return rres, rresQ

    def find_amp(self,amp_start, amp_stop, timeout, wf=12):
        self.select_waveform(wf)
        self.set('A_init_spin_up','0');
        self.set('A_sbc','0');
        self.set('EU_beam_conf','0');
        self.set('A_sbc','0')
        self.set('A_init_spin_up','0')
        self.set('EU_beam_conf','0')
        damp = (amp_stop-amp_start)
        for i in range(timeout):
            amp_ini = amp_start + damp*(1-np.exp(-(2*i/(timeout-1))))
            amp_fin = amp_ini + amp_ini/1000
            #print(amp_ini,amp_fin)

            name,data = self.run('EU_pulse_depth', start=amp_ini, stop=amp_fin, numofpoints=0, expperpoint=70, counter=None, live_plots_flag=False, verbose=False)
            #print(data)
            prb = data[0][1][0]
            ref = data[1][1][0]
            ctr = prb/ref
            #print(amp_ini, ctr)
            if ctr<.7:
                print('Lets use amp = ',  0.85*amp_ini)
                return 0.85*amp_ini
            if self.count_ions()>0:
                break
        return 0.

    def pulseRF_detBDx(self, amp_ini, amp_fin, wf=4, numofpoints=11, expperpoint=100):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf)
        self.set('A_sbc','0')
        self.set('A_init_spin_up','0')
        self.set('EU_beam_conf','0')
        if self.count_ions()>0:
            name,data = self.run('EU_pulse_depth', start=amp_ini, stop=amp_fin, numofpoints=numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=False)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();

    def std_SBC(self, Nions=1, mode=0):
        if Nions==1 and mode==1:
            self.set('A_sbc','1')
            self.set('no_sbc_outer','5')
            self.set('no_sbc_inner_1R','15')
            self.set('no_sbc_inner_2R','7')
            self.set('cool_calc_Q','0')
            self.set('cool_mode_lf','1')
            self.set('cool_RCKlf','0')
            self.set('cool_mode_mf','0')
            self.set('cool_mode_hf','0')
        if Nions==1 and mode==2:
            self.set('A_sbc','1')
            self.set('no_sbc_outer','3')
            self.set('no_sbc_inner_1R','17')
            self.set('no_sbc_inner_2R','0')
            self.set('cool_calc_Q','0')
            self.set('cool_mode_lf','0')
            self.set('cool_RCKlf','0')
            self.set('cool_mode_mf','1')
            self.set('cool_mode_hf','1')
        elif Nions==1 and mode==0:
            self.set('A_sbc','1')
            self.set('no_sbc_outer','5')
            self.set('no_sbc_inner_1R','13')
            self.set('no_sbc_inner_2R','5')
            self.set('cool_calc_Q','0')
            self.set('cool_mode_lf','1')
            self.set('cool_RCKlf','0')
            self.set('cool_mode_mf','1')
            self.set('cool_mode_hf','1')

    def pulseRF_detROC(self, amp_ini, amp_fin, wf=4, numofpoints=11, expperpoint=100):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf)
        self.std_SBC()
        self.set('A_init_spin_up','1')
        self.set('EU_beam_conf','20')
        if self.count_ions()>0:
            name,data = self.run('EU_pulse_depth', start=amp_ini, stop=amp_fin, numofpoints=numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=True)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();

    def pulseRF_detMF(self, amp_ini, amp_fin, cool_lf=1, wf=4, numofpoints=11, expperpoint=100):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf)
        self.set('A_sbc','1')
        self.set('A_init_spin_up','0')
        self.std_SBC()
        self.set('cool_mode_lf',str(cool_lf))
        self.set('EU_beam_conf','21')
        if self.count_ions()>0:
            name,data = self.run('EU_pulse_depth', start=amp_ini, stop=amp_fin, numofpoints=numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=True)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();
            print(name)

    def pulseRF_flopMF(self, amp, wf=4, numofpoints=5, expperpoint=100):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf, amp=amp)

        #self.set('amp',str(amp))
        self.std_SBC()
        self.set('EU_beam_conf','21')
        dur=float(self.get('t_mf_1R'))
        stop=4.5*dur
        if self.count_ions()>0:
            name,data = self.run('EU_flop', start=0, stop=stop, numofpoints=4.5*numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=True)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();
            print(name)

    def pulseRF_op_twbtw(self, amp, wf=4, numofpoints=5, expperpoint=100):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf, amp=amp)
        self.set('A_sbc','1')
        self.std_SBC()
        self.set('EU_beam_conf','21')
        t_wbtw=float(self.get('t_wbtw'))
        fr_mf=float(self.get('fr_mf'))

        start=t_wbtw-0.25/fr_mf
        stop=t_wbtw+0.25/fr_mf

        if self.count_ions()>0:
            self.single_fit_position('EU_two_pulses_t_wbtw', 't_wbtw', invert=False, give_all_para=False, start=start, stop=stop, expperpoint=expperpoint, numofpoints=numofpoints, verbose=True)

    def pulseRF_op_2ndAmp(self, amp, Delta=0.075, wf=4, numofpoints=5, expperpoint=100):
        self.select_waveform(wf, amp=amp)
        self.set('A_sbc','1')
        self.std_SBC()
        self.set('EU_beam_conf','21')
        Damp2=float(self.get('Damp2'))

        start=Damp2-0.5*Delta
        stop=Damp2+0.5*Delta

        if self.count_ions()>0:
            self.single_fit_position('EU_two_pulses', 'Damp2', invert=False, give_all_para=False, start=start, stop=stop, expperpoint=expperpoint, numofpoints=numofpoints, verbose=True)

    def pulseRF_scan_twbtw(self, amp, start=0, stop=0.3, wf=4, numofpoints=25, expperpoint=50):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf, amp=amp)
        self.set('A_sbc','1')
        self.std_SBC()
        self.set('EU_beam_conf','21')
        if self.count_ions()>0:
            name,data = self.run('EU_two_pulses_t_wbtw', start=start, stop=stop, numofpoints=numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=True)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();
            print(name)

    def EU_op_ML_offset(self, amp, delta=0.025, wf=4, numofpoints=12, expperpoint=50):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf, amp=amp)
        self.std_SBC()
        self.set('EU_beam_conf','21')
        self.set('A_init_spin_up','1')
        offset=float(self.get('offset'))
        if self.count_ions()>0:
            self.single_fit_position('EU_misc', 'offset', invert=False, give_all_para=False, par_name='offset', start=offset-delta/2, stop=offset+delta/2, numofpoints=numofpoints, expperpoint=expperpoint, live_plots_flag=False)

    def EU_MotionalSE_amp(self, start=0, stop=0.7, wf=4, numofpoints=25, expperpoint=50):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf)
        self.std_SBC()
        self.set('EU_beam_conf','21')
        offset=float(self.get('offset'))
        if self.count_ions()>0:
            name,data = self.run('EU_MSE_amp', par_name='amp', start=start, stop=stop, numofpoints=numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=True)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();
            print(name)

    def pulseRF_pulseRF_flopMF(self, amp, wf=4, numofpoints=7, expperpoint=100):
        #name,_ = self.run('EU_pulse_depth', start=0.0, stop=0.5, numofpoints=9, expperpoint=50)
        self.select_waveform(wf, amp=amp)
        self.set('A_sbc','1')
        self.std_SBC()
        self.set('EU_beam_conf','21')
        dur=float(self.get('t_mf_1R'))
        stop=6.*dur
        if self.count_ions()>0:
            name,data = self.run('EU_two_pulses_flop', start=0, stop=stop, numofpoints=6*numofpoints, expperpoint=expperpoint, counter=None, live_plots_flag=True)
            self.plot_data(data,name)
            plt.show()
            self.check_ion();
            print(name)

    def pcm_scan(self, delta, profile_name, shim_name, N_pts=25, N_avg=5, N_drop=1, verbose=False):
        pcm = PCM(self.results_path)

        init_value, last_date = self.get_profile(profile_name,shim_name)
        print(init_value, last_date)
        parameters = np.linspace(init_value-delta,init_value+delta,N_pts)
        time_start = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

        hists = []
        hists_err = []
        stats = []
        stats_err = []
        rate = []
        t = np.array([])
        try:
            for i,v in enumerate(parameters):
                print('%i/%i'%(i+1,N_pts))
                self.update_shim(profile_name,shim_name,v)
                for j in range(N_drop):
                    pcm.get_raw()
                # record N_avg points for averaging and deviation estimates
                l_h = np.zeros((0,36))
                l_arr = np.zeros((0,5))
                l_rate = np.zeros((0,1))
                for j in range(N_avg):
                    if verbose:
                        print('\t%i/%i'%(j+1,N_avg))
                    n,t,bins,nrm,arr,R = pcm.get_analyse(verbose=verbose)
                    bins = np.reshape(bins,(1,-1))
                    l_h = np.append(l_h, bins, axis=0)
                    l_arr = np.append(l_arr, arr, axis=0)
                    l_rate = np.append(l_rate, np.reshape(R,(1,1)), axis=0)
                l_h_avg = np.mean(l_h,axis=0)
                l_h_err = np.std(l_h,axis=0)/np.sqrt(l_h.shape[0])
                l_arr_avg = np.mean(l_arr,axis=0)
                l_arr_err = np.std(l_arr,axis=0)/np.sqrt(l_arr.shape[0])
                hists.append(l_h_avg.tolist())
                hists_err.append(l_h_err.tolist())
                stats.append(l_arr_avg.tolist())
                stats_err.append(l_arr_err.tolist())
                rate.append(l_rate.tolist())

            time_stop = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.update_shim(profile_name,shim_name,init_value)
            storage = pcm.pack(time_start, time_stop, profile_name, shim_name, init_value, parameters, t, hists, hists_err, stats, stats_err, rate)
            pcm.plot_stats(storage)
            fname = pcm.save(storage)

            return fname
        except KeyboardInterrupt:
            self.update_shim(profile_name,shim_name,init_value)
            return None

    def pcm_scan_3D(self, deltaX, deltaY, deltaZ, numofpoints=9, expperpoint=5, N_drop=1, verbose=False):
        namex = self.pcm_scan(deltaX, 'EU_cool', 'ExExpZ', N_pts=numofpoints, N_avg=expperpoint, N_drop=N_drop, verbose=verbose)
        namey = self.pcm_scan(deltaY, 'EU_cool', 'EyExpZ', N_pts=numofpoints, N_avg=expperpoint, N_drop=N_drop, verbose=verbose)
        namez = self.pcm_scan(deltaZ, 'EU_cool', 'EzExpZ', N_pts=numofpoints, N_avg=expperpoint, N_drop=N_drop, verbose=verbose)
        return namex,namey,namez


################################################ new function von Florian - working

    def tickle_Ramsey(self, mode_name, amp, dur, t_Ramsey, expperpoint=100, numofpoints=9, rnd_sampled=False, **kwargs):
        if self.count_ions()>0:
            fr=float(self.get('fr_'+mode_name))
            #print(fr)
            mode_name_upper = mode_name.upper()
            script = 'PDQ_'+mode_name_upper+'_Ramsey'
            print(script)
            self.set('t_tickle_'+mode_name,str(dur));
            self.set('u_tickle_'+mode_name,str(amp));
            self.set('t_ramsey_tickle',str(t_Ramsey));
            start=-.65
            stop=.65
            popt, perr = [0,0,0,0,0], [0,0,0,0,0]
            val, err, R2, popt, perr, chi2 = self.single_fit_phase_pi(script, 'dummy', invert=True, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, live_plots_flag=False,  give_all_para=True, rnd_sampled=rnd_sampled, **kwargs)
            return popt[0], perr[0]
        else:
            print('No Ions')



    def tickle_Ramsey_coherence(self, mode_name, amp, dur, expperpoint=100, numofpoints=12, rnd_sampled=False, iter_start=0, iter_end=1000, iter_steps=4, tickle_before=True, prec=0.5,**kwargs):
        cal_begin = self.print_header('Starting a Tickle Ramsey Coherence Experiment')

        #lbl = ['Ramsey dur. (µs)','Contrast (a.u.)','Mode #'+mode_name, 'Model: Decay w. exp']
        if tickle_before == True:
                if mode_name == 'lf':
                    self.set('tickle_shim_lf','ERaO');
                    self.cal_mode_fr('PDQ_'+str(mode_name.upper())+'_FScan',mode_name, amp/prec, dur*prec,expperpoint=50,numofpoints=15)
                if mode_name == 'mf':
                    self.set('tickle_shim_mf','EZO_fine');
                    self.cal_mode_fr('PDQ_'+str(mode_name.upper())+'_FScan',mode_name, amp/prec, dur*prec,expperpoint=50,numofpoints=15)
                if mode_name == 'hf':
                    self.set('tickle_shim_hf','EZO_fine');
                    self.cal_mode_fr('PDQ_'+str(mode_name.upper())+'_FScan',mode_name, amp/prec, dur*prec,expperpoint=50,numofpoints=15)
                time.sleep(0.05)

        A_liste = []
        A_err_liste = []
        t_liste = []
        iter_ = sorted(np.linspace(iter_start, iter_end, iter_steps), key = lambda x: random.random())
        A,A_err = self.tickle_Ramsey(mode_name=mode_name, amp=amp, dur=dur, t_Ramsey=0.0, numofpoints=numofpoints, expperpoint=expperpoint, rnd_sampled=rnd_sampled, **kwargs)
        A_liste.append(A)
        A_err_liste.append(A_err)
        t_liste.append(0.0)
        for i in range(len(iter_)):
            A,A_err = self.tickle_Ramsey(mode_name=mode_name, amp=amp, dur=dur, t_Ramsey=iter_[i], numofpoints=numofpoints, expperpoint=expperpoint, rnd_sampled=rnd_sampled, **kwargs)
            A_liste.append(A)
            A_err_liste.append(A_err)
            t_liste.append(iter_[i])

            try:
                guess = [iter_end/2.,1.]
                data = [np.array(t_liste), np.array(A_liste)/A_liste[0], np.array(A_err_liste)/A_liste[0]]
                popt, perr, R2, chi2 = self.fit_data(func_decay_exponential, data, guess, plt_labels=['Ramsey duration (µs)','Norm. Contrast', 'Ramsey on mode: '+mode_name, 'Exp. decay'])
            except:
                print('No fit!')

        popt_r,perr_r = round_sig(popt,perr)
        print(self.prnt_rslts('Coherence time of '+mode_name+' mode in µs',popt[0],perr[0]))
        self.set('t_Ramsey_coherence_'+str(mode_name),str(popt_r[0]))
        self.set('t_ramsey_tickle',str(0))
        self.print_footer(cal_begin)

        return t_liste, A_liste, A_err_liste

    ###############################################

    def mw_Ramsey(self,transition='3p3_2p2',t_ramsey=0,
                            numofpoints=10,expperpoint=100,rnd_sampled=False,
                           **kwargs):
        transition = '9_'+str(transition)+'_Ramsey_Phase'
        self.set('t_ramsey',str(t_ramsey))
        start=0
        stop=2

        popt, perr = [0,0,0,0,0], [0,0,0,0,0]
        val, err, R2, popt, perr, chi2 = self.single_fit_phase(transition, 'phase', invert=False, expperpoint=expperpoint, numofpoints=numofpoints, start=start, stop=stop, live_plots_flag=False,  give_all_para=True, rnd_sampled=rnd_sampled,**kwargs)
        return popt[0],perr[0]

    def mw_Ramsey_coherence(self,transition='3p3_2p2',
                            iter_start=0, iter_end=100, iter_steps=5,
                            numofpoints=22, expperpoint=100,rnd_sampled=False,
                           **kwargs):
        cal_begin = self.print_header('Starting a MW Ramsey Coherence Experiment')


        A_liste = []
        A_err_liste = []
        t_liste = []
        iter_ = sorted(np.linspace(iter_start, iter_end, iter_steps), key = lambda x: random.random())
        A,A_err = self.mw_Ramsey(transition=transition,t_ramsey=0.0, numofpoints=numofpoints,expperpoint=expperpoint,rnd_sampled=rnd_sampled,
                               **kwargs)
        A_liste.append(A)
        A_err_liste.append(A_err)
        t_liste.append(0.0)

        for i in range(len(iter_)):
            A,A_err = self.mw_Ramsey(transition=transition,t_ramsey=iter_[i], numofpoints=numofpoints,expperpoint=expperpoint,rnd_sampled=rnd_sampled,
                               **kwargs)
            A_liste.append(A)
            A_err_liste.append(A_err)
            t_liste.append(iter_[i])

            try:
                x = np.array(t_liste)
                y = np.array(A_liste/A_liste[0])
                yerr = np.array(A_err_liste/A_liste[0])
                data = [x,y,yerr]
                self.do_plot = True
                guess = [iter_end/2,1]
                popt, perr, R2, chi2 = self.fit_data(func_decay_exponential, data, guess, plt_labels=['Ramsey duration (µs)','Norm. Contrast', 'Ramsey on trans. '+transition, 'Exp. decay'], plot_residuals=True)
            except:
                print('no fit')

        #print(popt[0],perr[0])
        print(self.prnt_rslts('Coherence time of transition '+transition+' in µs',popt[0],perr[0]))
        self.set('t_coherence_'+transition,popt[0])
        self.print_footer(cal_begin)
        self.set('t_ramsey',str(0))
        #return t_liste


    def session_replay_exp(self,session_id):
        data = self.session_replay(session_id)
        A_liste = []
        A_err_liste = []
        for i in range(len(data)):
            A_liste.append(data[i][3][0])
            A_err_liste.append(data[i][4][0])



    def heating_rates(self, t_heat= 0., mode_name='lf', set_dur=False, numofpoints=13, expperpoint=15, counter=None, invert=[False,True], give_all_para=True, name_var='', xlabel='x', ylabel='y', add_contrast=False, rnd_sampled=True, **kwargs):
        self.session_start();

        self.set('t_heat',str(t_heat))
        if mode_name == 'lf':
            fr = float(self.get('fr_lf_1R'))
            fr_name = 'fr_lf_1R'
            dur = float(self.get('t_lf_1R'))
            script_name = '1_1R_LF_MA_fr'

        elif mode_name == 'mf':
            fr = float(self.get('fr_mf_1R'))
            dur = float(self.get('t_mf_1R'))
            script_name = 'Rad1_1R_MF_MA_fr'

        elif mode_name == 'hf':
            fr = float(self.get('fr_hf_1R'))
            dur = float(self.get('t_hf_1R'))
            script_name = 'Rad1_1R_HF_MA_fr'
        else:
            print('Undefined mode selector!')

        width = 2/dur
        fr_0 = fr - width/2.
        fr_1 = fr + width/2.

        self.do_plot = False
        name, data = self.run_single(script_name, start=fr_0, stop=fr_1, \
                        numofpoints=numofpoints, expperpoint=expperpoint, rnd_sampled=rnd_sampled, \
                        live_plots_flag=False, counter=None, verbose=False)

        title = [str(script_name)+' - Blue Side Band - t_heat = '+str(t_heat)+' ms',
                str(script_name)+' - Red Side Band - t_heat = '+str(t_heat)+' ms']

        ret_list = []
        for i,(d,inv) in enumerate(zip(data,invert)):
            if i==counter or counter is None:
                ret = self.single_fit_func(d, self.fit_single_sinc, invert=inv, give_all_para=True, name_var=name_var, xlabel='Frequency (MHz)', ylabel='Counts', add_contrast=False, text=None, title=None)
                ret_list.append(ret)
                self._log('fit %s %s %s %s %i %i %s'%('heating_rates', name, 'dummy', self.fit_single_sinc.__name__, inv, i, title[i]))

        n,n_err = self.fit_simultaneously_freq(data, ret_list, set_dur=set_dur, dur=dur)

        print(self.prnt_rslts('n',n,n_err))
        sid = self.session_stop()
        return n,n_err,sid



    def fit_simultaneously_freq(self, data, ret_list,  dur=10, t_heat = None, set_dur = False, plot_together=True):
        #dur = 10 is an arbitrary value, just that the other functions still working, without changing them!
        x = [data[0][0],data[1][0]]
        y = [data[0][1],data[1][1]]
        yerr = [data[0][2],data[1][2]]

        if ret_list==None:
            self.do_plot = False
            ret_list = []
            counter=None
            invert = [False,True]
            for i,(d,inv) in enumerate(zip(data,invert)):
                ret = self.single_fit_func(d, self.fit_single_sinc, invert=inv, give_all_para=True, name_var='', xlabel='Frequency (MHz)', ylabel='Counts', add_contrast=False, text=None, title=None)
                ret_list.append(ret)

        x_data = np.hstack([x[0],x[1]])
        y_data = np.hstack([y[0],y[1]])
        yerr_data = np.hstack([yerr[0],yerr[1]])
        if set_dur == True:
            sigmabr = 1./dur
            #Set abitrary value for the error!
            sigmabr_err = 0.01
            def poly(x_, Ab, Ar, x0br, Cb, Cr):
                #all this is just to split x_data into the original parts of x
                l= len(x[0])
                l1= len(x[1])
                s=l+l1
                #Exchange Ab with n, Ab = Ar/n+Ar
                #This leads to enormes errors!
                t = np.hstack([
                Ab*np.sinc(((x_[:l]-x0br)/sigmabr))**2+Cb,
                -Ar*np.sinc(((x_[l:(s)]-x0br)/sigmabr))**2-Cr
                ])
                return t

            guess = [ret_list[0][3][1], ret_list[1][3][1], ret_list[0][3][3], ret_list[0][3][0], ret_list[1][3][0]]

            (Ab, Ar, x0br, Cb, Cr), pcov_poly_fit = curve_fit(poly, x_data, y_data, p0=guess,sigma=yerr_data)
            Ab_err = np.sqrt(pcov_poly_fit[0][0])
            Ar_err = np.sqrt(pcov_poly_fit[1][1])
            x0br_err = np.sqrt(pcov_poly_fit[2][2])
            Cb_err = np.sqrt(pcov_poly_fit[3][3])
            Cr_err = np.sqrt(pcov_poly_fit[4][4])

        if set_dur == False:
            def poly(x_, Ab, Ar, x0br, sigmabr, Cb, Cr):
                #all this is just to split x_data into the original parts of x
                l= len(x[0])
                l1= len(x[1])
                s=l+l1
                #Exchange Ab with n, Ab = Ar/n+Ar
                #This leads to enormes errors!
                t = np.hstack([
                Ab*np.sinc(((x_[:l]-x0br)/sigmabr))**2+Cb,
                -Ar*np.sinc(((x_[l:(s)]-x0br)/sigmabr))**2-Cr
                ])
                return t

            guess = [ret_list[0][3][1], ret_list[1][3][1], ret_list[0][3][3], ret_list[0][3][2], ret_list[0][3][0], ret_list[1][3][0]]
            (Ab, Ar, x0br, sigmabr, Cb, Cr), pcov_poly_fit = curve_fit(poly, x_data, y_data, p0=guess,sigma=yerr_data)
            Ab_err = np.sqrt(pcov_poly_fit[0][0])
            Ar_err = np.sqrt(pcov_poly_fit[1][1])
            x0br_err = np.sqrt(pcov_poly_fit[2][2])
            sigmabr_err = np.sqrt(pcov_poly_fit[3][3])
            Cb_err = np.sqrt(pcov_poly_fit[4][4])
            Cr_err = np.sqrt(pcov_poly_fit[5][5])

        if Ar<0:
            Ar = -Ar
        if Ar>Ab:
            print('Ar > Ab -> BREAK!')
            #return 0,0
        R = Ar/Ab
        R_err = R*np.sqrt((Ab_err/Ab)**2+(Ar_err/Ar)**2)
        n = R/(1.-R)
        n_err = 1./((1.-R)**2)*R_err

        n_exact = n
        n_exact_err = n_err

        arg_names_list = ['n','Ab','Ar','x0br','sigmabr','Cb','Cr']
        popt_list = [n, Ab, Ar, x0br, sigmabr, Cb, -Cr]
        popt_err_list = [n_err,Ab_err,Ar_err,x0br_err,sigmabr_err,Cb_err,Cr_err]
        #sincSquare(x, C, A, sigma, mu)
        R2_b = self.cal_R2(data[0][0],data[0][1], [popt_list[5],popt_list[1],popt_list[4],popt_list[3]])
        R2_r = self.cal_R2(data[1][0],data[1][1], [popt_list[6],-popt_list[2],popt_list[4],popt_list[3]])


        if plot_together == True:
            z = np.linspace(np.min(x[0]),np.max(x[0]),100)

            f, ax = plt.subplots()
            ax.errorbar(x[0], y[0], yerr[0], fmt='o', color = 'C0', ecolor='C7', elinewidth=1, capsize=3, label = 'blue sideband')
            ax.errorbar(x[1], y[1], yerr[1], fmt='o', color = 'C3', ecolor='C7', elinewidth=1, capsize=3, label = 'red sideband')
            ax.plot(z, Ab*np.sinc(((z-x0br)/sigmabr))**2+Cb, 'C0-', linewidth=3)
            ax.plot(z, -Ar*np.sinc(((z-x0br)/sigmabr))**2-Cr, 'C3-', linewidth=3)
            ax.set_xlabel('Frequency (MHz)')
            ax.set_ylabel('Counts')
            mw_contr_high = float(self.get('mw_contr_high'))
            mw_contr_low = float(self.get('mw_contr_low'))
            ax.axhline(y=mw_contr_high,ls='--',c='grey')
            ax.axhline(y=mw_contr_low,ls='--',c='grey')
            #HERE IS THE R2 Missing
            txt = r'   $R^2_b$ = %.2f%%'%(R2_b*100) + '\n'+r'   $R^2_r$ = %.2f%%'%(R2_r*100) + '\n'
            for m,v,e in zip(arg_names_list, popt_list, popt_err_list):
                n = max(significant_digit(e),0)
                s_v = '%%.%if'%(n)
                txt += '%5s'%m + ' '+r'= ' + s_v%v
                if np.isfinite(e):
                    txt += '(%i)\n'%(e*10**n)
                else:
                    txt += '(inf)\n'
            txt = txt[:-1]

            ax.text(1.02, .78, txt, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
            f.suptitle('Merged graph of both side bands with same x0 and sigma', fontsize=10)
            plt.grid()
            #ax.legend()
            plt.show()
        return n_exact,n_exact_err


    def cal_R2(self,xdata,ydata,popt):
        residuals = ydata - sincSquare(xdata,*popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata-np.mean(ydata))**2)
        R2 = 1 - (ss_res / ss_tot)
        return R2

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


    def heating_rates_iteration(self, mode_name='lf',iter_start=0, iter_end=20, iter_steps=2, numofpoints=11 ,expperpoint=100,rnd_sampled=False, random=False, zero=False, **kwargs):
        #cal_begin=self.print_header('Axial heating rate')
        name_var = 'nbar_per_sec_1R_' + mode_name

        iter_ = np.linspace(iter_start, iter_end, iter_steps)
        if random == True:
            random.shuffle(iter_)
        nbar_liste = []
        nbar_err_liste = []
        t_heat_liste = []
        sid_liste = []
        sid_and_t_heat_liste = []
        iter_ = np.linspace(iter_start, iter_end, iter_steps)
        for i in range(len(iter_)):
            nbar,nbar_err,sid = self.heating_rates(mode_name=mode_name, t_heat=iter_[i], numofpoints=numofpoints, expperpoint=expperpoint, rnd_sampled=rnd_sampled,**kwargs)
            nbar_liste.append(nbar)
            nbar_err_liste.append(nbar_err)
            t_heat_liste.append(iter_[i])
            sid_and_t_heat = str(sid)+'_'+str(iter_[i])
            sid_and_t_heat_liste.append(sid_and_t_heat)
            try:
                x = np.array(t_heat_liste)
                y = np.array(nbar_liste)
                yerr = np.array(nbar_err_liste)
                data = [x,y,yerr]
                self.do_plot = True
                ret_lin = self.single_fit_func(data, self.fit_linear, invert=False, add_contrast=False, xlabel='t_heat in ms', ylabel='average n', give_all_para=True, plot_residuals=True)
            except:
                print('no fit')
            if zero == True:
                self.heating_rates(mode_name=mode_name, t_heat=0, numofpoints=5, expperpoint= 5, rnd_sampled=True,**kwargs)

        m,m_err = ret_lin[3][0]*1000, ret_lin[4][0]*1000
        print(self.prnt_rslts('Heating rate (quanta/s)',m,m_err))

        #in seconds!
        self.set(name_var,str(m));
        self.set('t_heat',str(0))


        #To get every sid into one entry of the log:
        sid_join = []
        for j in range(len(sid_and_t_heat_liste)):
            for i in range(len(sid_and_t_heat_liste[j])):
                sid_join.append(sid_and_t_heat_liste[j][i])
            sid_join.append('+')

        sid_join = ''.join(sid_join)

        self.session_start()
        self._log('fit %s %s %s %s %i %i %s'%('heating_rates_iteration', sid_join, 'dummy', 'poly+lin', False, 0, 'Heating rates iteration'))
        sid = self.session_stop()

        return sid

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

### Build scripts

    def create_eios_script(self, seq = ['PDQ_init','COOL_d', 'DET_bdx'], verbose=True, s_hlp=False):
        pb_scrpt_prts_pd=pd.read_csv("../UserData/script_lines_2020_02_25.csv", index_col=0)

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

    def create_scan_para(self, par_type='fr', name='lf', scl=1., start=None, stop=None, npts=None, nexp=None, fit_func=None, invert=None):
        if par_type=='fr':

            if name == 'tickle_lf':
                par_name=par_type+'_lf'
            elif name == 'tickle_mf':
                par_name=par_type+'_mf'
            elif name == 'tickle_hf':
                par_name=par_type+'_hf'
            else:
                par_name=par_type+'_'+name
            cnt=float(self.get(par_name))
            if name == 'shift_sq_mf':
                name='sq_mf'
            dur=float(self.get('t_'+name))
            if start==None:
                start=cnt-scl*2/dur
            if stop==None:
                stop=cnt+scl*2/dur
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
            cnt=float(self.get('phs_'+name))
            if start==None:
                start=cnt-scl*np.pi
            if stop==None:
                stop=cnt+scl*np.pi
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
            nexp=100

        return par_name, start, stop, npts, nexp, fit_func, invert

    def set_ips(self, ips=None):
        if ips != None:
            for ip in ips:
                self.set(ip[0],str(ip[1]))

    def run_seq(self, seq=['COOL_d', 'DTCT_bdx'],
                par_type=None, par_name='dummy', scl=1.0, scl_pts=1., ips=None, start=None, stop=None, npts=None, nexp=None, rnd_sampled=False,
                fit_func=None, invert=None, fit_result=True, set_op_para=False, check_script=False):
        '''
        Define your experimental PAULBOX sequence and
        run it with default (or your choice of) parameters
        '''

        self.set_ips(ips=ips)
        script_name=self.create_eios_script(seq=seq, verbose=check_script);
        par_name, start, stop, npts, nexp, fit_func, invert=self.create_scan_para(par_type=par_type,name=par_name, scl=scl, start=start, stop=stop, npts=npts, nexp=nexp, fit_func=fit_func, invert=invert)
        npts=scl_pts*npts
        if check_script==True:
            print('Scan_para: %s \n Start: %.5f, \n Stop:: %.5f\n N_pts: %i \n N_exp: %i \n Fit_func: %s \n Fit_invert: %s'%(par_name, start, stop, npts, nexp, fit_func, invert))
           #print('s')
        if check_script==False:
            #self.set_ips(ips=ips)
            if self.get('spin_up_mw')=='1' and invert==True:
                invert=False
            else:
                if self.get('spin_up_mw')=='1' and invert==False:
                    invert=True
            cntr=-1
            descr='Experimental sequence:'
            for pulse in seq:
                if pulse=='header' or pulse=='footer':
                    descr+='\n'
                elif pulse=='DTCT0_bdx' or pulse=='DTCT1_bdx' or pulse=='PDQ_init':
                    if pulse=='DTCT0_bdx'and cntr==-1:
                        cntr=0
                    if pulse=='DTCT1_bdx'and cntr==-1:
                        cntr=1
                    if pulse=='DTCT1_bdx' and cntr==0:
                        cntr=None
                    if pulse=='DTCT0_bdx' and cntr==1:
                        cntr=None
                    descr+='\n *'+pulse+'\n ------'
                else:
                    descr+='\n *'+pulse
            if set_op_para==True:
                name_var = par_name
            else:
                name_var=''
            #print(cntr)

            self.do_live=False
            text_ips='IonProps:';
            for i in ips:
                text_ips+='\n'+str(i)
            print(text_ips)
            if fit_func==None or fit_result==False:
                ret = self.run(script_name=script_name, par_name=par_name, start=start, stop=stop, numofpoints=npts, expperpoint=nexp, rnd_sampled=rnd_sampled, counter=cntr)
                name, data = ret
                if cntr!=None:
                    data = [data]
                self.plt_raw_dt(data=data, name=name, xlabel=par_name+' (a.u.)', ylabel='Cts.', text=descr)
            else:
                ret = self.single_fit_run(script_name=script_name, par_name=par_name, start=start, stop=stop, numofpoints=npts, expperpoint=nexp, invert=invert, xlabel=par_name+' (a.u.)', ylabel='Cts.', name_var=name_var, add_contrast=True, give_all_para=True, func=fit_func, text=descr, counter=cntr, rnd_sampled=rnd_sampled)

            self.do_live=False
            return ret
        else:
            if par_type==None:
                return 'Fit not run...', [[0.0],[0.0],[0.0]]
            else:
                return 0.0, 0.0, 0.0, [[0.0],[0.0],[0.0]], [[0.0],[0.0],[0.0]], [0.0]

    def plt_raw_dt(self, data, name, xlabel='Scan_par (a.u.)', ylabel='Cts.', text=''):
            high_fluo=float(self.get('mw_contr_high'))
            low_fluo=float(self.get('mw_contr_low'))
            if len(data)>0:
                fig, ax = plt.subplots()
                ColorList=['navy','red','orange','grey','silver','black']
                if name:
                    fig.canvas.set_window_title(name)
                for i,data in enumerate(data):
                    x,y,y_err = data
                    plt.errorbar(x, y, yerr = y_err, linestyle = "None", marker = "o", label='Det.# %i'%i, color=ColorList[i], markersize=7.5, lw=1., capsize=.0);
                plt.legend(loc='upper right')
            plt.title(name[-23:])
            if text is not None:
                    plt.text(1.02, .78, text, va="center", ha="left", bbox=dict(alpha=0.3), transform=ax.transAxes)
            plt.axhline(high_fluo, color='gray', ls='--')
            plt.axhline(low_fluo, color='gray', ls='--')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            #plt.show()

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
        ips_pd.to_csv('../UserData/ips.csv')
        if loc != None:
            print(ips_pd.loc[loc: loc+'zzz', ['Setting']])
        return ips_pd

    def ana_mf_sq(self, name, fix=[0,0,0,0,1,1,0], nmax = 35, tangle = 38.0):
        from PyModules.analyse_eios.eios_data import read, read_xml, find_files
        from PyModules.analyse_eios.eios_sb import open_file, LDparameter, fit_RSB_BSB_par,fit_flop_sb,plot_flop_fit, fit_RSB_BSB
        cache_path='./data/'
        def print_ips(name, ips):
            _, root = read(name)
            ip = read_xml(root)['ionproperties']
            #print('File:', name)
            print('----------------')
            print('Ion properties:')
            print('----------------')
            ip_l = []
            for i in ips:
                print(i+':', ip[i])
                ip_l.append(ip[i])
            print('----------------')
            return ip_l
        redflop, blueflop, lbl = open_file([name], cache_path)
        ip_l = print_ips(name, ['fr_mf','t_tickle_mf_EU', 'u_tickle_mf_EU', 'rabi_mf', 'n_th_mf'])
        f_mf = ip_l[0]
        LD = LDparameter(f_mf,tangle)
        print(LD)
        Rabi_init = 0.270
        dec_init = 0.017
        limb_init = 0.5
        limr_init = 0.9
        nth = 0.1
        ncoh = 0.000
        nsq = 0.0
        inits = [Rabi_init, dec_init, limb_init, limr_init, nth, ncoh, nsq]
        red_chi, fmin, param = fit_RSB_BSB_par(redflop, blueflop, LD, nmax, inits, fix, ntrot=1, doplot = True)
        #red_chi, fmin, param, m, flop_func_list, value, error, fit_fockdist_norm, [fit_fock_n, fit_fock_p, fit_fock_e] = fit_flop_sb(redflop, blueflop, LD, nmax, inits, fix, ntrot = 1)
        #[fit_rabi, fit_dec, fit_limb, fit_limr, fit_nth, fit_ncoh, fit_nsq] = value
        #[fit_rabi_err,fit_dec_err,fit_limb_err,fit_limr_err,fit_nth_err,fit_ncoh_err,fit_nsq_err] = error
        #fit_status = '$red. \chi^2$= %.3f\n $\Omega_{0}$= %.3f +- %.3f\n $\Gamma_{dec}$= %.3f +- %.3f\n $n_{th}$= %.3f +- %.3f\n$n_{coh}$= %.3f +- %.3f\n$n_{sq}$= %.3f +- %.3f' % (red_chi, fit_rabi, fit_rabi_err, fit_dec, fit_dec_err, fit_nth, fit_nth_err, fit_ncoh, fit_ncoh_err, fit_nsq, fit_nsq_err)
        #plot_flop_fit(flop_func_list, fit_fock_n, fit_fock_p, fit_fock_e, [redflop, blueflop], lbl, fit_status, figsize=(10,4));
        #plt.show();
        return ip_l, red_chi, fmin, param

    def check_photo_freq(self, freq=525.40575, verbose=True):
        from PyModules.wavemeter.lock_client import web_lock_client
        wlc = web_lock_client(host='10.5.78.145', port=8000)
        Photo_wmch = 'WMCH3'
        wlc.activate(Photo_wmch)
        gt_trc_Photo=wlc.get_trace_last(Photo_wmch)
        time.sleep(.15)
        wlc.deactivate(Photo_wmch)
        txt='--------------------------\n'
        txt+='Check Photo wavelength:\n'
        txt+='--------------------------\n'
        txt+='Photo freq. (THz): '+str(round((gt_trc_Photo[1]['trace']),5))+'\n'
        str_ret='Photo det. (MHz): '+str(round((freq-gt_trc_Photo[1]['trace'])*1e6,1))+'\n'
        txt+=str_ret
        txt+='--------------------------'
        if verbose == True:
            print(txt)
        return str_ret
