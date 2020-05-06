#!/usr/bin/env python3

import sys
sys.path.insert(0, './../')

import matplotlib.pyplot as plt

from time import sleep
from threading import Thread, Lock
from PyModules.ipc_client import IPC_client

ipc_data_sep = '\x1c'; # ASCII 28: file separator
ipc_cmd_sep = '\x1e'; # ASCII 30: record separator

#run script, ttl ME/PB, trigger PB, get/set IP
class EIOS_META_SCRIPT:
    def __init__(self, srv_adr_cmd = '/tmp/socket-eios-cmd', srv_adr_data = '/tmp/socket-eios-data'):
        try:
            self.client_com = IPC_client(srv_adr_cmd)
        except:
            raise Exception("Socket %s could not be found! Check EIOS!" % (srv_adr_cmd))
        try:
            self.client_data = IPC_client(srv_adr_data)
        except:
            raise Exception("Socket %s could not be found! Check EIOS!" % (srv_adr_data))
        self.queue_dict = {}
        self.run_live = True
        self.th_live = Thread()

        self.data = {}
        self.mutex = Lock()

    def __del__(self):
        try:
            del self.client_data
            del self.client_com
            self.run_live = False
            if self.th_live.is_alive():
                self.th_live.join()
        except AttributeError or NameError:
            #pass
            raise

    def _parse_data(self,ret):
        s_id = -1; name = ''; data = '';
        data_split = ret.split(ipc_data_sep)
        n = len(data_split)
        if n==3: # warning: n>4 = overflow
            s_id = int(data_split[0])
            name = data_split[1]
            data = data_split[2]
        else: # message fragmented
            print(n, data_split)
            raise Exception('Received message fragmented %i'%n)
        data_sets = data.split('\n\n')
        '''
        print('**********************************************')
        print(data_sets)
        print('______________________________________________')
        '''
        data_ctr = []
        for i,data_cnt in enumerate(data_sets):
            points = data_cnt.split('\n')

            x = []; y = []; y_err = []
            if len(points)>0:
                if len(points[0])>0:
                    #print(points[0])
                    if points[0][0]=='#':
                        #y_err = int(points[0][1:])
                        points = points[1:]

            for p in points:
                sp = p.split('\t')
                if len(sp)>1:
                    x.append(float(sp[0]))
                    y.append(float(sp[1]))
                if len(sp)>2:
                    y_err.append(float(sp[2]))
            if len(x)>0:
                data_ctr.append([x,y,y_err])
        return s_id, name, data_ctr

    def receive_data(self):
        ret = self.client_data.receive()
        return len(ret)>0, ret

    def query(self, cmd):
        with self.mutex:
            ret = self.client_com.query('%s%c'%(cmd,ipc_cmd_sep))
            data_split = ret.split(ipc_cmd_sep)
            msg = data_split[0]
            if len(msg)>1:
                return bool(float(ret[0])),msg[1:]
            elif len(msg)>0:
                return bool(float(ret[0])),""
            else:
                return False,""

    def _live(self):
        while(self.run_live and self.queue_dict):
            #print('_live',[key for key,_ in self.queue_dict.items()], self.run_live)
            #print([key for key,_ in self.queue_dict.items()], self.run_live, ' ', end='')
            stat,ret = self.receive_data()
            #print(stat, "%r"%ret)
            if stat:
                msgs = ret.split(ipc_cmd_sep)
                for msg in msgs:
                    if msg:
                        s_id, name, data_ctr = self._parse_data(msg)
                        func = None
                        if s_id in self.queue_dict:
                            func = self.queue_dict[s_id]
                        #print(s_id in self.queue_dict, func is not None, bool(name), len(name))
                        if func is not None:
                            last_call = bool(name)
                            recall = func(s_id, data_ctr, last_call = last_call)
                            #print(bool(name),func,last_call,recall)
                            if (not recall) or last_call:
                                del self.queue_dict[s_id]

    def _add_queue(self, s_id, func):
        self.data[s_id] = []
        self.queue_dict[s_id] = func
        if not self.th_live.is_alive():
            self.run_live = True
            self.th_live = Thread(target=self._live, args=())
            self.th_live.start()

    def _update_data(self, s_id, data_ctr, last_call):
        if last_call:
            self.data[s_id] = data_ctr.copy()

    def _live_wrapper(self, s_id, data_ctr, last_call, func=None):
        self._update_data(s_id, data_ctr, last_call)
        if func is not None:
            return func(s_id, data_ctr, last_call)
        else:
            return True

    def set(self,name,value):
        """
        set(name, value)

        Set ion property.

        Sets the ion property `name` to string `value`.

        Parameters
        ----------
        name : string
            Ion property name.

        value : string
            Alpha numeric value as string.

        Returns
        -------
        stat : bool
            Success status.
        """
        stat,ret = self.query("ip %s %s" % (name,value))
        return stat

    def get(self,name):
        """
        get(name)

        Get ion property.

        Returns the value of ion property `name` as string.

        Parameters
        ----------
        name : string
            Ion property name.

        Returns
        -------
        ret : string
            Alpha numeric value as string if successful, else "".
        """
        stat,ret = self.query("ip %s" % (name))
        if stat:
            return ret
        else:
            return ""

    def set_profile(self,profile_name,shim_name,value):
        """
        set_profile(profile_name, shim_name, value)

        Set shim profile.

        Sets `shim_name` in `profile_name` to string `value`.

        Parameters
        ----------
        profile_name : string
            Profile name.

        shim_name : string
            Shim name.

        value : string
            Numeric value as string.

        Returns
        -------
        stat : bool
            Success status.
        """
        stat,ret = self.query("profile %s %s %s" % (profile_name,shim_name,value))
        return stat

    def get_profile(self,profile_name,shim_name):
        """
        get_profile(profile_name, shim_name)

        Get shim profile value.

        Returns string value of `shim_name` in `profile_name`.

        Parameters
        ----------
        profile_name : string
            Profile name.

        shim_name : string
            Shim name.

        Returns
        -------
        ret : string
            Numeric value as string if successful, else "".
        """
        stat,ret = self.query("profile %s %s" % (profile_name,shim_name))
        return ret

    def ttl_ME(self, channel, state=None):
        """
        ttl_ME(channel, state=None)

        Set/get TTL `channel` of Meilhaus card.

        If state is None: return ttl state
        Else: set ttl to `state`

        Parameters
        ----------
        channel : int
            Channel index: 0 to 7.

        state : bool, None, optional
            Set ttl to `state` if not None. Default is None

        Returns
        -------
        ret : bool, int
            Retruns success or ttl state if `state`==None.
        """
        if state is None:
            stat,ret = self.query("ttlme %u" % (channel))
            if stat:
                return int(ret)
            else:
                return -1
        else:
            stat,ret = self.query("ttlme %u %i" % (channel,state))
            return stat

    def ttl_PB(self,channel,state=None):
        """
        ttl_PB(channel, state=None)

        Set/get TTL `channel` of Paulbox.

        If state is None: return ttl state
        Else: set ttl to `state`

        Parameters
        ----------
        channel : int
            Channel index: 0 to 15.

        state : bool, None, optional
            Set ttl to `state` if not None. Default is None

        Returns
        -------
        ret : bool, int
            Retruns success or ttl state if `state`==None.
        """
        if state is None:
            stat,ret = self.query("ttlpb %u" % (channel))
            if stat:
                return int(ret)
            else:
                return -1
        else:
            stat,ret = self.query("ttlpb %u %i" % (channel,state))
            return stat

    def trigger(self,channel,t_width):
        stat,ret = self.query("trigger %u %i" % (channel,t_width))
        return stat

    def pdq_trigger(self,trigger_cnt = 1): # 1 = one trigger, 2 = two trigger
        s_id,name,_ = self.run("pdq_trigger", p_name='dummy', p_start=0, p_stop=1, p_numofpoints=trigger_cnt-1, p_expperpoint=1, rnd_sampled=0, live_plots_flag=0)
        return (s_id>-1), name

    def set_queue(self, run=True, interleaved=False, random=False):
        """
        set_queue(run=True, interleaved=False, random=False)

        Set EIOS queue parameter: Run, Interleaved, Random

        Parameters
        ----------
        run : bool, optional
            Run/stop scripts execution. Default is True.

        interleaved : bool, optional
            Scripts run interleaved. Default is False.

        random : bool, optional
            Running scripts get experiment time randomly assigned. Default is False.

        Returns
        -------
        stat : bool
            Success status.
        """
        stat,ret = self.query("queue %i %i %i"%(run, interleaved, random))
        return stat

    def _run_cmd(self, cmd, block=False, func=None, live_wrapper=False):
        s_id = -1; name = ''; data_ctr = [];
        try:
            stat,ret = self.query(cmd)
            if stat:
                s_id, name, _ = self._parse_data(ret)
                if block:
                    if live_wrapper:
                        func_wrapper = lambda s_id, data_ctr, last_call: self._live_wrapper(s_id, data_ctr, last_call, func=func)
                        self._add_queue(s_id,func_wrapper)
                    else:
                        self._add_queue(s_id,self._update_data)
                    try:
                        while (s_id in self.queue_dict):
                            sleep(.1)
                    except KeyboardInterrupt as e:
                        self.stop()
                        while (s_id in self.queue_dict):
                            sleep(.1)
                    data_ctr = self.data[s_id].copy()
                    del self.data[s_id]
                    return s_id, name, data_ctr
                else:
                    self._add_queue(s_id,func)
                    return s_id, name
        except KeyboardInterrupt as e:
            self.queue_dict = {}
            self.stop()
            raise KeyboardInterrupt('Stop all running scripts!')
        except BrokenPipeError as e:
            raise EiosPipeError('Connection to EIOS lost: check EIOS and restart kernel!')
        except:
            raise
        return s_id, name

    def run(self, script_name, p_name=None, \
            p_start=None, p_stop=None, \
            p_numofpoints=None, p_expperpoint=None, \
            rnd_sampled=None, live_plots_flag=None):
        """
        run(script_name, p_name=None, \
            p_start=None, p_stop=None, \
            p_numofpoints=None, p_expperpoint=None, \
            rnd_sampled=None, live_plots_flag=None)

        Run (blocking) script in EIOS and return result file name and data.
        The default parameters depend on the selected script. See EIOS GUI for detailes.

        Parameters
        ----------
            script_name : string
                Name of script.

            p_name : string, optional
                Scan parameter (variable).

            p_start, p_stop : float, optional
                Start and stop parameter of scan interval.

            p_numofpoints : int, optional
                Number of points in interval.

            p_expperpoint : unsigned int
                Experiments per point.

            rnd_sampled : bool
                Measurement order of points in interval are randomly selected.

            live_plots_flag : bool
                Show live measurement in EIOS GUI (gnuplot). Does not affect Python (live) plots.

        Returns
        -------
        s_id : int
            Script id.

        name : string
            Full path to measurement file.

        data_ctr : list of counter
            List of counter = [x,y,y_err], with x, y, y_err list of floats


        """
        s_id = -1; name = '';
        if None in [p_name, p_start, p_stop, p_numofpoints, p_expperpoint, rnd_sampled, live_plots_flag]:
            cmd = 'run %s' % script_name
        else:
            if p_numofpoints<0:
                return s_id,name
            cmd = 'run %s %s %f %f %u %u %u %u' % (script_name, p_name, p_start, p_stop, p_numofpoints, p_expperpoint, rnd_sampled, live_plots_flag)
        return self._run_cmd(cmd, block=True, live_wrapper=False)


    def add(self, script_name, func, p_name=None, p_start=None, p_stop=None, p_numofpoints=None, p_expperpoint=None, rnd_sampled=None, live_plots_flag=None, block=False):
        """
        add(script_name, p_name=None, \
            p_start=None, p_stop=None, \
            p_numofpoints=None, p_expperpoint=None, \
            rnd_sampled=None, live_plots_flag=None,
            block = False)

        Same as run, but blocking is optional.
        Run (non-)blocking script in EIOS and return result file name and data.
        The default parameters depend on the selected script. See EIOS GUI for detailes.

        Parameters
        ----------
            script_name : string
                Name of script.

            func : callable function, func(s_id, data_ctr, last_call)
                This function is called on every update of data points and when the script is finished/aborted (last_call=True).
                s_id : script id
                data_ctr : data list
                last_call : script ended

            p_name : string, optional
                Scan parameter (variable).

            p_start, p_stop : float, optional
                Start and stop parameter of scan interval.

            p_numofpoints : int, optional
                Number of points in interval.

            p_expperpoint : unsigned int
                Experiments per point.

            rnd_sampled : bool
                Measurement order of points in interval are randomly selected.

            live_plots_flag : bool
                Show live measurement in EIOS GUI (gnuplot). Does not affect Python (live) plots.

            block : bool
                Scripts can run in parallel. Optionally blocking or non-blocking.
                When a script is finished, func with last_call=True is called.
                And if True, return includes data_ctr

        Returns
        -------
        s_id : int
            Script id.

        name : string
            Full path to measurement file.

        [data_ctr : list of counter], optional -> only if block is True
            List of counter = [x,y,y_err], with x, y, y_err list of floats
        """
        s_id = -1; name = '';
        if None in [p_name, p_start, p_stop, p_numofpoints, p_expperpoint, rnd_sampled, live_plots_flag]:
            cmd = 'add %s' % script_name
        else:
            if p_numofpoints<0:
                return s_id,name
            cmd = 'add %s %s %f %f %u %u %u %u' % (script_name, p_name, p_start, p_stop, p_numofpoints, p_expperpoint, rnd_sampled, live_plots_flag)
        return self._run_cmd(cmd, block=block, func=func, live_wrapper=True)

    def run_me(self, channels, sampling=100, average=1, block=False, func=None, show_plots_flag=False):
        """
        run_me(channels, sampling=100, average=1, block=False, func=None, show_plots_flag=False)

        Records `channels` of Meilhaus ADC with `average`s and `sampling` interval.

        Parameters
        ----------
            channels : int, list of int
                Channel index: 0 to 31.

            sampling : int, optional
                Number of milliseconds between samples. Default is 100.

            average : int, optional
                Count of averages. Default is 1.

            block : bool
                Python is blocking until measurement is aborted.
                When a script is finished, func with last_call=True is called.

            func : callable function, func(s_id, data_ctr, last_call)
                This function is called on every update of data points and when the script is finished/aborted (last_call=True).
                s_id : script id
                data_ctr : data list
                last_call : measurement ended

            show_plots_flag : bool
                Show live measurement in EIOS GUI (gnuplot). Does not affect Python (live) plots.

        Returns
        -------
        s_id : int
            Script id.

        name : string
            Full path to measurement file.

        [data_ctr : list of counter], optional -> only if block is True
            List of counter = [x,y,y_err], with x, y, y_err list of floats
        """
        live_wrapper = (func is not None) and callable(func)
        send_data_flag=True
        stream_data_flag=live_wrapper
        live_plots_flag=show_plots_flag

        s_id = -1; name = '';
        if isinstance(channels, int):
            ch_str = str(channels)
        elif isinstance(channels, list):
            ch_str = ','.join(str(x) for x in channels)
        else:
            return s_id,name
        cmd = 'run_me %s %u %u %u %u %u' % (ch_str, sampling, average, live_plots_flag, send_data_flag, stream_data_flag)
        return self._run_cmd(cmd, block=block, func=func, live_wrapper=live_wrapper)

    def get_parameter(self,script_name):
        """
        get_parameter(script_name)

        Returns the parameters of `script_name`.

        Parameters
        ----------
            script_name : string
                Channel index: 0 to 31.

        Returns
        -------
        p_name : string
            Name of scan parameter.

        p_start, p_stop : float
            Start and stop of scan interval.

        p_numofpoints, p_expperpoint : int
            Number of points and experiments per point.

        random_sampled : int
            Points in interval are measured in random order

        shim_scan, shim_number, profile_number : int
            Is this a shim scan. What shim in which profile.

        """
        cmd = 'parameter %s' % script_name
        stat,ret = self.query(cmd)
        if not stat:
            return []
        par = ret.split(" ")
        if len(par)<9:
            print(par)
            print(len(par))
            raise Exception("Parameter count error: not enough parameter to unpack!")
        p_name = par[0]
        p_start = float(par[1])
        p_stop = float(par[2])
        p_numofpoints = int(par[3])
        p_expperpoint = int(par[4])
        random_sampled = int(par[5])
        shim_scan = int(par[6])
        shim_number = int(par[7])
        profile_number = int(par[8])
        return p_name, p_start, p_stop, p_numofpoints, p_expperpoint, random_sampled, shim_scan, shim_number, profile_number

    def list_script(self):
        """
        list_script()

        List all scripts.

        Returns
        -------
        ret : list of strings
            List of all scripts.

        """
        cmd = 'list_script'
        stat,ret = self.query(cmd)
        if not stat:
            return []
        return ret.split("\n")

    def list_ip(self):
        """
        list_ip()

        List all ion properties.

        Returns
        -------
        ret : list of strings
        """
        cmd = 'list_ip'
        stat,ret = self.query(cmd)
        if not stat:
            return []
        return ret.split("\n")

    def sourcecode(self, script_name):
        """
        list_ip()

        Return sourcecode of `script_name`.

        Returns
        -------
        ret : string
        """
        cmd = 'sourcecode %s' % script_name
        stat,ret = self.query(cmd)
        if not stat:
            return ''
        return ret

    def read_adc_me(self,channel):
        """
        read_adc_me(channel)

        Reads `channel` of Meilhaus card ADC.

        Parameters
        ----------
        channel : int
            Channel index: 0 to 31.

        Returns
        -------
        ret : float
            Analog voltage of `channel`.
            None if failed.
        """
        cmd = 'adcme %i' % channel
        stat,ret = self.query(cmd)
        if not stat:
            return float('nan')
        return float(ret)

    def get_dac_me(self,channel):
        """
        get_dac_me(channel)

        Reads current setting of DAC `channel` of Meilhaus card.

        Parameters
        ----------
        channel : int
            Channel index: 0 to 3.

        Returns
        -------
        ret : float
            Analog voltage of `channel`.
            None if failed.
        """
        cmd = 'dacme %i' % channel
        stat,ret = self.query(cmd)
        if not stat:
            return float('nan')
        return float(ret)

    def set_dac_me(self,channel,value):
        """
        set_dac_me(channel, value)

        Reads current setting of DAC `channel` of Meilhaus card.

        Parameters
        ----------
        channel : int
            Channel index: 0 to 3.

        value : float
            Voltage of DAC: -10V to 10V

        Returns
        -------
        ret : float
            Analog voltage of `channel`.
            None if failed.
        """
        cmd = 'dacme %i %f' % (channel,value)
        stat,ret = self.query(cmd)
        if not stat:
            return float('nan')
        return float(ret)

    def mirror_reset(self,idx):
        """
        mirror_reset(idx)

        Resets current position of Physik Instrumente mirror `idx` to 0. Mirror does not move.

        Parameters
        ----------
        idx : int
            Index of mirror. Depends on number of connected mirrors.

        Returns
        -------
        stat : bool
            Success status.
        """
        cmd = 'mirror_reset %i' % (idx)
        stat,ret = self.query(cmd)
        if not stat:
            print('Mirror error: %s'%ret)
        return stat

    def get_mirror_pos(self,idx):
        """
        get_mirror_pos(idx)

        Return position of Physik Instrumente mirror `idx`.

        Parameters
        ----------
        idx : int
            Index of mirror. Depends on number of connected mirrors.

        Returns
        -------
        ret : float
            Current absolute mirror position in steps.
        """
        cmd = 'mirror_pos %i' % (idx)
        stat,ret = self.query(cmd)
        if not stat:
            print('Mirror error: %s'%ret)
            ret = 'nan'
        return float(ret)

    def mirror_pos(self,idx,pos):
        """
        mirror_pos(idx, pos)

        Move Physik Instrumente mirror `idx` to absolute position `pos`.

        Parameters
        ----------
        idx : int
            Index of mirror. Depends on number of connected mirrors.

        pos : int
            New absolute mirror position.

        Returns
        -------
        stat : bool
            Success status.
        """
        cmd = 'mirror_pos %i %i' % (idx,pos)
        stat,ret = self.query(cmd)
        if not stat:
            print('Mirror error: %s'%ret)
        return stat

    def mirror_stp(self,idx,stp):
        """
        mirror_stp(idx, pos)

        Move Physik Instrumente mirror `idx` by `stp` steps.

        Parameters
        ----------
        idx : int
            Index of mirror. Depends on number of connected mirrors.

        stp : int
            Move mirror by `stp` steps (positive or negtive).

        Returns
        -------
        stat : bool
            Success status.
        """
        cmd = 'mirror_stp %i %i' % (idx,stp)
        stat,ret = self.query(cmd)
        if not stat:
            print('Mirror error: %s'%ret)
        return stat

    def status(self, s_id=None):
        """
        status(s_id)

        Returns status of currently running script(s).
        If s_id is None, status of all scripts is returned.

        Parameters
        ----------
        s_id : int
            Script index or None. Default is None.

        Returns
        -------
        res : dict
            res[id(s)] = {'name':string, 'progress':float, 'file':string}
        """
        if (s_id is None):
            cmd = 'status'
        else:
            cmd = 'status %i' % s_id
        stat,ret = self.query(cmd)
        res = {}
        if stat:
            for s in ret.split(';'):
                th_info = s.split(',')
                if len(th_info)==4:
                    key, name, prog, file = th_info
                    res[int(key)] = {'name':name, 'progress':float(prog), 'file':file}
        return res

    def stop(self,s_id=None):
        """
        stop(s_id)

        Stops script with s_id.
        If s_id is None, all running scripts are aborted.

        Parameters
        ----------
        s_id : int
            Script index or None. Default is None.

        Returns
        -------
        stat : bool
            Success status.
        """
        if (s_id is None):
            cmd = 'stop'
        else:
            cmd = 'stop %i' % s_id
        stat,ret = self.query(cmd)
        return stat

    def end(self):
        """
        end()

        Stops IPC.

        Returns
        -------
        stat : bool
            Success status.
        """
        stat,ret = self.query('end')
        return stat

    def exit(self):
        """
        exit()

        Closes EIOS.

        Returns
        -------
        stat : bool
            Success status.
        """
        stat,ret = self.query('exit')
        return stat

    def plot_data(self,name,data_ctr, lbls=['','']):
        """
        plot_data(name, data_ctr)

        Plots EIOS counter data.

        Parameters
        ----------
        name : string
            Window title of plot.

        data_ctr : list of [counter = [x, y, y_err]]

        Returns
        -------
        fig : matplotlib figure object
        """
        if len(data_ctr)>0:
            #fig = plt.figure()
            fig, ax = plt.subplots()
            ColorList=['navy','red','orange','grey','silver','black']
            if name:
                fig.canvas.set_window_title(name)
            for i,data in enumerate(data_ctr):
                x,y,y_err = data
                plt.errorbar(x, y, yerr = y_err, linestyle = "None", marker = "o", label='CNT %i'%i, color=ColorList[i], markersize=7.5, lw=1., capsize=.0,);
            plt.legend(loc='upper right')
            #plt.grid(linestyle='--')
            #plt.grid(b=True, which='major', axis='both')
            plt.title(name)
            plt.ylim(bottom=-0.05)
            plt.xlabel(lbls[0])
            plt.ylabel(lbls[1])
            #plt.xlim(left=0.)
            #plt.draw()
            #print('flag')
            import datetime
            fname='./data/scans/'+str(datetime.datetime.now())[:22].replace(' ','_').replace(':','-').replace('.','-')+'.png'
            plt.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.75)
            return fig
        else:
            return None

import os
class trigger:
    def __init__(self):
        self.eios = EIOS_META_SCRIPT()
        self.execute = False

    def __del__(self):
        self.stop()
        del self.eios

    def run(self,t_sleep,remove=True):
        self.execute = True
        ret = True
        while (self.execute and ret):
            sleep(t_sleep)
            ret,fname = self.eios.pdq_trigger()
            if ret and remove:
                try:
                    os.remove(fname)
                except OSError as e:
                    print('Measurement file "%s" could not be removed: OSError %i' % (fname,e.errno))

    def stop(self):
        self.execute = False


if __name__ == "__main__":
    from time import sleep

    eios = EIOS_META_SCRIPT()

    if len(sys.argv)>1:
        cmd = sys.argv[1]
        pos = cmd.find("run ")
        if ( pos > -1):
            script_name = cmd[pos+4:]
            s_id,name,data = eios.run(script_name)
            eios.plot_data(data,'%i %s'%(s_id,name))
            plt.show()
        else:
            ret = eios.query(cmd)
            print(ret)
    else:

        id_s, name = eios.add('BDX',None,'dummy',0.,100.,2000,200,False,False)
        while(True):
            info = eios.status()
            if id_s in info:
                print('%.2f%%'%(info[id_s]['progress']*100))
            else:
                print('not running')
            sleep(2)

        #print(eios.status(6))
        #print(eios.sourcecode('BDX'))
        '''
        callback = lambda s_id, data_ctr: not eios.stop(s_id)
        sid, name = eios.add('BDX',callback,'dummy',0.,100.,2000,200,False,False)
        '''
        '''
        def live_plt(s_id, data_ctr, last_call=False):
            #plt.scatter(data_ctr[0][0], data_ctr[0][1])
            #plt.pause(0.05)
            #print(s_id)
            bar = [' ']*80
            t = data_ctr[0][0][-1]
            y = data_ctr[0][1][-1]
            #Y = max(data_ctr[0][1])
            Y = 12.
            j = (y/Y)*len(bar)
            bar[max(0,min(int(j),len(bar)-1))] = 'x'
            print('%05.2f\t%s'%(t, "".join(bar)))
            return True
        sid, name = eios.add('BDX',live_plt,'dummy',0.,100.,2000,200,False,False)
        print(sid, name)
        '''
        '''
        sid, name, data = eios.run('BDX','dummy',0.,100.,2000,200,False,False)
        print(sid, name, data)
        '''

        #for ip in eios.list_ip():
        #    print('%s = %s'%(ip,eios.get(ip)))
        #print(eios.list_script())
        #print(eios.set_queue(True,True,False))
        #print(eios.get_profile(profile_name='EU_squeeze',shim_name='ch3p8'))
        #print(eios.set_profile(profile_name='EU_cool',shim_name='ch3p8',value='3.141'))
        #print(eios.exit())
        #print(eios.end())
        #print(eios.get_parameter("BDD"))
        #print(eios.pdq_trigger(1))
        #print(eios.set_queue(1,0,0))
