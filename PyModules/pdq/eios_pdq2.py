#!/usr/bin/python3

from serial import Serial
from .pdq2 import Pdq2
from serial.tools.list_ports import comports as listcomports
from math import sqrt, fmod, copysign, cos, sin, pi

import json
#DEBUG:
#print(json.dumps(dictionary, indent = 4))

#need to avoid numpy, scipy
from scipy import interpolate

import os
import sys

# reduce error stack to only the upper most error
#sys.tracebacklimit = 0

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
# Global Variable, once device names (like DREIECK0, DREIECK1) have been translated to devices
#  they will be stored here and not be looked up again. the lookup took ~14ms every datapoint!
this.device_list = None

#for colorfull console output :) c.f. https://stackoverflow.com/a/287944
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def device_from_serial(serial_number):
    device = None
    for dev in listcomports():
        if dev.serial_number == serial_number:
            print("PDQ device %s found @ port %s" % (dev.serial_number, dev.device) )
            device = dev.device
    return device

def phase_modulo(phase): #set the phase to the range (-0.5, 0.5]
    return (phase + 0.5) % 1 - 0.5

class EIOS_PDQ2():
    softlimit = 9.5
    hardlimit = 9.999

    def __init__(self, serial_numbers, voltages, multiplier = False, channelsperdac = 9, reset_on_write = False):

        if (this.device_list is None):
            this.device_list = list()
            for i, serial_number in enumerate(serial_numbers):
                device = device_from_serial(serial_number)
                if (device is not None ):
                    this.device_list.append(device)
                else:
                    print("PDQ device %s could NOT be found!" % serial_number)

        freq = 50e6
        if multiplier:
            freq *= 2.
        self.clock_period = 1./freq
        
        self._channelsperdac = channelsperdac
        self._numstacks = 0
        self._pdq_stacks = []
        voltages = self.voltagelimit(voltages)
        for i, device in enumerate(this.device_list):
            stack_voltages = voltages[i*channelsperdac : (i+1)*channelsperdac]
            self._pdq_stacks.append( EIOS_PDQ2_STACK(device, stack_voltages, multiplier, channelsperdac, reset_on_write) )
            self._numstacks += 1
        #print(serial_numbers, voltages, multiplier, channelsperdac, self._numstacks)

    def pdq_init(self, voltages):
        if (self._numstacks>0):
            voltages = self.voltagelimit(voltages)
            for i in range(self._numstacks):
                self._pdq_stacks[i].pdq_init(voltages[i*self._channelsperdac:(i+1)*self._channelsperdac])
            
    def pdq_step_to(self, tstep, voltages, special=False, **kwargs):
        if (self._numstacks>0):
            voltages = self.voltagelimit(voltages)
            for i in range(self._numstacks):
                stack_voltages = voltages[i*self._channelsperdac:(i+1)*self._channelsperdac]
                self._pdq_stacks[i].pdq_step_to(tstep, stack_voltages, special, stepsize = 12, **kwargs)

    def pdq_chirp(self, amplitudes, start_freq, stop_freq, sweep_time, phases, points):
        if (self._numstacks>0):
            amplitudes = self.voltagelimit(amplitudes)
            for i in range(self._numstacks):
                stack_amp = amplitudes[i*self._channelsperdac:(i+1)*self._channelsperdac]
                stack_phase = phases[i*self._channelsperdac:(i+1)*self._channelsperdac]
                self._pdq_stacks[i]._pdq_chirp_start(amplitudes, start_freq, stop_freq, sweep_time, phases, points, phase_reset=False)
                self._pdq_stacks[i]._pdq_chirp_stop()

    def pdq_tickle_start(self, amplitudes, frequencies, phases, phase_reset=False, trigger=True):
        for i in range(self._numstacks):
            stack_amp = amplitudes[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_freq = frequencies[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_phase = phases[i*self._channelsperdac:(i+1)*self._channelsperdac]
            self._pdq_stacks[i].pdq_tickle_start(stack_amp, stack_freq, stack_phase, phase_reset, trigger)

    def pdq_tickle_stop(self):
        for i in range(self._numstacks):
            self._pdq_stacks[i].pdq_tickle_stop()

    def pdq_tickle_pulse(self, amplitudes, frequencies, pulse_width, phases, phase_reset, pulse_shape):
        for i in range(self._numstacks):
            stack_amp = amplitudes[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_freq = frequencies[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_phase = phases[i*self._channelsperdac:(i+1)*self._channelsperdac]
            self._pdq_stacks[i].pdq_tickle_pulse(stack_amp, stack_freq, pulse_width, stack_phase, phase_reset, pulse_shape)

    def pdq_tickle_offset(self, amplitudes, frequencies, duration, phases, phase_reset, offsets):
        for i in range(self._numstacks):
            stack_amp = amplitudes[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_freq = frequencies[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_phase = phases[i*self._channelsperdac:(i+1)*self._channelsperdac]
            stack_offsets = offsets[i*self._channelsperdac:(i+1)*self._channelsperdac]
            self._pdq_stacks[i].pdq_tickle_offset(stack_amp, stack_freq, duration, stack_phase, phase_reset, stack_offsets)

    def pdq_pulse(self, times, voltages, data_waveform, amp = 1., offset = 0., order=0, special=0, init_trigger=True):
        '''
            data_waveform.shape = (N_wf, N_spline, N_wf_points=N)
            data_waveform[channels][splines][step] = 
                    [channels: 
                        [splines: 
                                [  y_0, ..., y_n],
                                [dy1_0, ..., dy1_n],
                                ...,
                                [dy2_0, ..., dy2_n] 
                        ]
                    ]
        '''
        N_wf = len(data_waveform)
        N = len(times)
        dt = (times[1]-times[0])

        # Assignment of waveform data to channels specified in shim (voltages)
        thres_zero = 1e-5
        assign = []
        cntr = 0
        for i,v in enumerate(voltages):
            if v>thres_zero:        
                assign.append(cntr%N_wf)
                cntr += 1
            else:
                assign.append(-1)
        # Spline interpolation / reshaping of splines <-> time steps
        if order<0 or order>3:
            print(bcolors.FAIL + ("Valid interpolation order range from 0..3 (0: const, 1: lin, 2: quad, 3: cubic)") + bcolors.ENDC)
            raise ValueError("Only splines up to cubic order are supported.")
        # spline divieder
        divs = [ self.clock_period**j for j in range(order+1)]
        new_data = []
        for idx_wf, data in enumerate(data_waveform):
            #print('wf%i, N_sp %i, N_points %i'%(idx_wf,len(data),len(data[0])))
            if order and len(data)==1:
                print('Spline interpolation: order %i'%order)
                tck = interpolate.splrep(times, data[0], k=order, s=0)
                splines = interpolate.spalde(times, tck)
                # normalize coefficients to clock_period = 10 ns or 20 ns
                for i in range(len(splines)):
                    for j in range(1,len(splines[i])):
                        splines[i][j] *= divs[j]
            else:
                # Reshape e.g (4, 100) -> (100, 4)
                splines = [[ x[i] for x in data] for i in range(N)]

            # Apply offset correction
            if len(splines[0]) == 1:
                s0 = splines[0][0]
                for i in range(len(splines)):
                    splines[i][0] = amp*(splines[i][0]-s0) + s0 + offset
                    #splines[i][0] += offset
            new_data.append(splines)

        #print('assign', assign)
        #print('data', new_data)
        # Assign voltages & waveform channels to pdq stack
        for i in range(self._numstacks):
            idx_start = i*self._channelsperdac
            idx_stop = (i+1)*self._channelsperdac
            stack_amp = voltages[idx_start:idx_stop] # apply amp correction
            stack_assign = assign[idx_start:idx_stop]
            self._pdq_stacks[i].pdq_pulse(N, dt, stack_amp, stack_assign, new_data, special, init_trigger)

    def pdq_write(self):
        for i in range(self._numstacks):
            try:
                self._pdq_stacks[i].pdq_write()
            except:
                #print("Error:", sys.exc_info()[0])
                raise

    def single_voltagelimit(self, voltage, it, softlimit, hardlimit):
        if abs(voltage) > hardlimit:
            new_voltage = copysign(hardlimit,voltage)
            print(bcolors.FAIL + ("Voltage[%02i] = %gV not in range ]-10V,10V[, set to hard limit %gV!" % (it,voltage,new_voltage)) + bcolors.ENDC)
            voltage = new_voltage
        elif abs(voltage) > softlimit:
            print(bcolors.WARNING + ("Warning: Voltage[%02i] =  %gV (soft limit %gV)" % (it,voltage,copysign(self.softlimit,voltage)) ) + bcolors.ENDC)
        return voltage

    def voltagelimit(self, voltages):
        for i in range(len(voltages)):
            voltages[i] = self.single_voltagelimit(voltages[i],i,self.softlimit,self.hardlimit)
        return voltages

'''
def check_voltage(voltage):
    value = abs(voltage)
    #sign = voltage/value
    if value>softlimit:
        if value>hardlimit:
            return False
            #voltage = sign*hardlimit
    #return voltage
    return True
    
def calc_voltage(step,steps,a0,a1,a2,a3):
    dt = (step/steps)
    return a0+a1*dt+2.*a2*(dt**2)+6.*a3*(dt**3)

def check_voltage_calc(steps,a0,a1,a2,a3):
    steps = int(steps)
    volt = [calc_voltage(x,steps,a0,a1,a2,a3) for x in range(steps+1)]
    if not check_voltage(max(volt)):
        raise AttributeError("Maximum voltage 10V exceeded!")
'''

class EIOS_PDQ2_STACK():
    def __init__(self, device, voltages, multiplier = False, num_channels = 9, reset_on_write = False):
        self._device = device
        if self._device is not None:
            self.freq = 50e6
            self.multiplier = multiplier
            if multiplier:
                self.freq *= 2.
            self.dac_divider = 1./self.freq
            self._numchannels = num_channels
            self._program = []
            self._zero_list = [0]*self._numchannels
            # == np.zeros(self._numchannels) so we wont need numpy :)
            self._last_frequencies = self._zero_list
            self._last_phases = self._zero_list
            self._min_ttl_length = 0.6e-6
            self._ttl_steps = round(self._min_ttl_length / self.dac_divider)
            self.pdq_init(voltages)
            self.reset = reset_on_write

    def _pdq_bias(self, voltages, duration, trigger=False, wait=True):
        segment = {'trigger': trigger, 'duration': duration, 'channel_data':list() }
        for c in range(self._numchannels):
            single_channel_data = {'bias': {'amplitude':[voltages[c]], 'wait': wait}}
            segment['channel_data'].append(single_channel_data)
        self._program.append(segment)

    def _pdq_const(self, voltages, trigger=False, wait=True):
        self._pdq_bias(voltages, duration=self._ttl_steps, trigger=trigger, wait=wait)
        self._last_voltages = voltages

    def pdq_init(self, voltages):
        self._pdq_const(voltages, trigger=True, wait=True)

    def pdq_step_to(self, tstep, voltages, special = 0, stepsize = 12, **kwargs):
        '''
        Makes a voltage step on all channels from current values
        (36-vector, _last_voltages, in V) to new values (voltages)
        in the duration tstep (in s) with a chosen ramp (special).

        Arguments:
        ----------
        tstep -- duration of the voltage step in s.
        voltages -- 36-vector of desired voltages after step.
        Keyword arguments:
        ------------------
        special -- chose between: 0 (linear slope, default),
        2 (cubic smoothstep), 3 (cubic smoothstep through specified
        lowpass filter, overshooting). Specify lowpass by
        resistance R (Ohms) and capicitance C (Farad).      
        '''

        
        steps = round(tstep / self.dac_divider)
        special = int(special)
        #print("steps: ", steps, "step size: ", stepsize)

        # #################### special 1 to 4 start #################### 
        if special < 5:
            segment = {"trigger": True, "duration":steps, "wait":True}
            channel_data = list()
            for c in range(self._numchannels):
                cvoltage = voltages[c]
                lvoltage = self._last_voltages[c]
                dvoltage = cvoltage - lvoltage

                if special == 0:
                    a0, a1 = linear_step(lvoltage, dvoltage, steps)
                    single_channel_data={"bias": {"amplitude":[a0, a1]}}
                elif special == 1:
                    a0, a1 = overshoot1(lvoltage, dvoltage, tstep)
                    #check_voltage_calc(steps,a0,a1,0.,0.)
                    single_channel_data={"bias": {"amplitude":[a0, a1]}}
                elif special == 2:
                    a0, a1, a2, a3 = smooth_step(lvoltage, dvoltage, steps)
                    single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3]}}
                elif special == 3:
                    a0, a1, a2, a3 = overshoot3(lvoltage, dvoltage, steps, self.dac_divider)
                    #check_voltage_calc(steps,a0,a1,a2,a3)
                    single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3]}}
                elif special == 4:
                    a0, a1, a2, a3 = overshoot4(lvoltage, dvoltage, steps, self.dac_divider)
                    #check_voltage_calc(steps,a0,a1,a2,a3)
                    single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3]}}

                channel_data.append(single_channel_data)

            segment['channel_data'] = channel_data
            self._program.append(segment)
            self._pdq_const(voltages, False)
        # #################### special 1 to 4 end #######################

        # #################### special 6 & 7 start ######################
        elif special == 6 or special == 7:
            bool_wait = (special == 7) # special 6: False; special 7: True
            self._pdq_const(voltages, trigger=(not bool_wait), wait=bool_wait)

        # #################### special 6 & 7 end ########################

        # #################### special 8 & 9 start ######################
        elif special == 8 or special == 9:
            bool_wait = (special == 9) # special 8: False; special 9: True
            segment = {"trigger": not bool_wait, "duration": steps}#wait here?
            #segment = {"trigger": True, "duration": steps}#wait here?
            channel_data = list()
            for c in range(self._numchannels):
                cvoltage = voltages[c]
                lvoltage = self._last_voltages[c]
                dvoltage = cvoltage - lvoltage
                a0, a1, a2, a3 = smooth_step(lvoltage, dvoltage, steps)
                single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3], "wait": bool_wait}}
                #single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3], "wait": False}}
                channel_data.append(single_channel_data)
            segment['channel_data'] = channel_data
            self._program.append(segment)
            self._last_voltages = voltages
        # #################### special 8 & 9 end ########################
   
        # #################### special 21 & 22 start #################### 
        elif special == 21 or special == 22:
            if kwargs:
                res = kwargs["a"]
                tme = kwargs["b"]
                par = kwargs["c"]
                capacitors = [2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9]
                resistors = [res[0],0.,res[1],res[2],0.,res[3],res[4],0.,res[5]]
                timedelays = [tme[0],0.,tme[1],tme[2],0.,tme[3],tme[4],0.,tme[5]]
            else:
                path = './UserData/Shims/overshootconfig.dat'
                print('Read from file: %s' % path)
                try:
                    timedelays = []
                    resistors = []
                    capacitors = []
                    with open(path, 'r') as f:
                        fiter = iter(f)
                        next(fiter)
                        for line in fiter:
                            columns = line.strip().split()
                            resistors.append(float(columns[0]))
                            capacitors.append(float(columns[1]))
                            timedelays.append(float(columns[2]))
                except IOError:
                   print('Could not open file!')
                   capacitors = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
                   resistors = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
                   timedelays = [0.,0.,0.,0.,0.,0.,0.,0.,0.]

            print("timedelays: ", timedelays)
            print("resistors: ", resistors)
            print("capacitors: ", capacitors)

            for step in range(1, int(steps)+1, stepsize):
                channel_data = list()
                # first step: this line waits for trigger
                if step == 1:
                    segment = {"trigger": True, "duration": stepsize, "wait":False}
                # last step: next line waits for trigger
                elif step > (steps - 2 * stepsize):
                    segment = {"trigger": False, "duration": steps - step, "wait":True}
                else:
                    segment = {"trigger": False, "duration": stepsize, "wait":False}
                for c in range(self._numchannels):
                    lv = self._last_voltages[c]
                    cv = voltages[c]
                    R0 = resistors[c]
                    C0 = capacitors[c] / self.dac_divider
                    tdelay = timedelays[c]
                    dv = cv - lv
                    a0, a1, a2, a3 = overshoot21(lv, dv, step, steps, tdelay, C0, R0)
                    #check_voltage_calc(steps,a0,a1,0.,0.)
                    single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3]}}
                    channel_data.append(single_channel_data)
                segment['channel_data'] = channel_data
                self._program.append(segment)
            self._pdq_const(voltages, False)
        # #################### special 21 & 22 end ####################

        # #################### special 23 start #######################
        elif special == 23:
            if kwargs:
                res = kwargs["a"]
                tme = kwargs["b"]
                par = kwargs["c"]
                capacitors = [2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9]
                resistors = [res[0],0.,res[1],res[2],0.,res[3],res[4],0.,res[5]]
                timedelays = [tme[0],0.,tme[1],tme[2],0.,tme[3],tme[4],0.,tme[5]]
                thold = par[0]
                tstepup = par[1]
                tstepdown = tstep
                stepsdown = round(tstepdown / self.dac_divider)
                ttotal = tstepdown + thold + tstepup
                stepshold = round(thold / self.dac_divider)
                stepsup = round(tstepup / self.dac_divider)
            else:
                print('Argument error: kwargs: a,b,c missing!')
                return

            # go down
            for step in range(1, int(stepsdown)+1, stepsize):
                channel_data = list()
                # first step: this line waits for trigger
                if step == 1:
                    segment = {"trigger": True, "duration": stepsize, "wait":False}
                # last step: next line waits for trigger, not here however
                elif step > (stepsdown - 2 * stepsize):
                    segment = {"trigger": False, "duration": stepsdown - step, "wait":False}
                else:
                    segment = {"trigger": False, "duration": stepsize, "wait":False}
                for c in range(self._numchannels):
                    lv = self._last_voltages[c]
                    cv = voltages[c]
                    R0 = resistors[c]
                    C0 = capacitors[c] / self.dac_divider
                    tdelay = timedelays[c]
                    dv = cv - lv
                    a0, a1, a2, a3 = overshoot21(lv, dv, step, stepsdown, tdelay, C0, R0)
                    #check_voltage_calc(steps,a0,a1,0.,0.)
                    single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3]}}
                    channel_data.append(single_channel_data)
                segment['channel_data'] = channel_data
                self._program.append(segment)
            # went down    
            # wait if stepshold > 12
            if stepshold >= stepsize:
                channel_data = list()
                segment = {"trigger": False, "duration": stepshold, "wait":False}
                for c in range(self._numchannels):
                    cv = voltages[c]
                    #check_voltage_calc(steps,a0,a1,0.,0.)
                    single_channel_data={"bias": {"amplitude":[cv]}}
                    channel_data.append(single_channel_data)
                    segment['channel_data'] = channel_data
                self._program.append(segment)
            # waited    
            # go up
            for step in range(1, int(stepsup)+1, stepsize):
                channel_data = list()
                # first step: this line waits for trigger, not here however
                if step == 1:
                    segment = {"trigger": False, "duration": stepsize, "wait":False}
                # last step: next line waits for trigger
                elif step > (stepsup - 2 * stepsize):
                    segment = {"trigger": False, "duration": stepsup - step, "wait":True}
                else:
                    segment = {"trigger": False, "duration": stepsize, "wait":False}
                for c in range(self._numchannels):
                    cv = self._last_voltages[c]
                    lv = voltages[c]
                    R0 = resistors[c]
                    C0 = capacitors[c] / self.dac_divider
                    tdelay = timedelays[c]
                    dv = cv - lv
                    a0, a1, a2, a3 = overshoot21(lv, dv, step, stepsup, tdelay, C0, R0)
                    #check_voltage_calc(steps,a0,a1,0.,0.)
                    single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3]}}
                    channel_data.append(single_channel_data)
                segment['channel_data'] = channel_data
                self._program.append(segment)    
            # went up                
            # set final voltages    
            #self._pdq_const(self._last_voltages, False)
        # #################### special 23 end ######################

        # #################### special 24 start #################### 
        elif special == 24:
            if kwargs:
                res = kwargs["a"]
                tme = kwargs["b"]
                par = kwargs["c"]
                capacitors = [2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9,2.2e-9]
                resistors = [res[0],0.,res[1],res[2],0.,res[3],res[4],0.,res[5]]
                timedelays = [tme[0],0.,tme[1],tme[2],0.,tme[3],tme[4],0.,tme[5]]
                thold = par[0]
                tstepup = par[1]
                tstepdown = tstep
                ttotal = tstepdown + thold + tstepup
                stepsdown = round(tstepdown / self.dac_divider)
                stepshold = round(thold / self.dac_divider)
                stepsup = round(tstepup / self.dac_divider)
            else:
                print('Argument error: kwargs: a,b,c missing!')
                return

            # go down
            segment = {"trigger": True, "duration": stepsdown}
            channel_data = list()
            for c in range(self._numchannels):
                cvoltage = voltages[c]
                lvoltage = self._last_voltages[c]
                dvoltage = cvoltage - lvoltage
                a0, a1, a2, a3 = smooth_step(lvoltage, dvoltage, stepsdown)
                single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3], "wait": False}}
                channel_data.append(single_channel_data)
            segment['channel_data'] = channel_data
            self._program.append(segment)
            # wait
            if stepshold >= stepsize:
                self._pdq_bias(voltages, stepshold, trigger=False, wait=False)
            # go up
            segment = {"trigger": False, "duration": stepsup}
            channel_data = list()
            for c in range(self._numchannels):
                lvoltage = voltages[c]
                cvoltage = self._last_voltages[c]
                dvoltage = cvoltage - lvoltage
                a0, a1, a2, a3 = smooth_step(lvoltage, dvoltage, stepsup)
                single_channel_data={"bias": {"amplitude":[a0, a1, a2, a3], "wait": True}}
                channel_data.append(single_channel_data)
            segment['channel_data'] = channel_data
            self._program.append(segment)            
            
            # set final voltages    
            #self._pdq_const(self._last_voltages, False)
        # #################### special 24 end ######################
            
        # #################### else: special not available #########
        else:
            print("Special %i is not available!" % special)

    def pdq_tickle_start(self, amplitudes, frequencies, phases, phase_reset=False, trigger=True):
        self._pdq_set_dds(self._ttl_steps, amplitudes, frequencies, phases, phase_reset, trigger)  
        
    def pdq_tickle_stop(self):
        self._pdq_set_dds(self._ttl_steps, self._zero_list, self._last_frequencies, self._last_phases, phase_reset=False, trigger=True)
        #self._pdq_const(self._last_voltages, trigger=True)

    def pdq_tickle_offset(self, stack_amp, stack_freq, duration, stack_phase, phase_reset, stack_offsets):
            steps = round(duration*1e-6 / self.dac_divider)
            #print(steps)
            # step to bias voltages
            #self._pdq_bias(stack_offsets, self._ttl_steps, trigger=True, wait=False)
            self._pdq_bias(stack_offsets, 1, trigger=True, wait=False)
            # start dds
            self._pdq_set_dds(steps, stack_amp, stack_freq, stack_phase, phase_reset, trigger=False)
            # stop dds
            self._pdq_set_dds(self._ttl_steps, self._zero_list, self._last_frequencies, self._last_phases, phase_reset=False, trigger=False)
            # step to old bias voltages
            self._pdq_bias(self._last_voltages, self._ttl_steps, trigger=False, wait=True)

    def pdq_tickle_pulse(self, amplitudes, frequencies, pulse_width, phases, phase_reset, pulse_shape):
        if type(pulse_shape) is list:
            l = len(pulse_shape)
            freq = frequencies[0]*1e-6 # MHz
            pdq_freq = self.freq*1e-6 # MHz
            pulse_width = int(pulse_width*pdq_freq)/pdq_freq # fitting to pdq sampling

            N = pulse_width*freq # number of periodes in pulse_width
            N_antinode = int(2.*N) # number of antinodes in pulse_width
            dt = 1./N_antinode # in units of pulse_width

            stps_pdq = int(pdq_freq*pulse_width*dt)
            dt = stps_pdq/(pdq_freq*pulse_width)

            interp_data = [[[pulse_shape[i%l],0,0,0],dt] for i in range(N_antinode)]
            interp_data.append([[0.,0., 0., 0.],0])
        else:
            if pulse_shape == 0:
                interp_data = interp_data_rect
            elif pulse_shape == 1:
                interp_data = interp_data_gaus
            elif pulse_shape == 2:
                interp_data = interp_data_sinc
            else:
                interp_data=interp_data_rect
        self._pdq_dds_pulse(amplitudes, frequencies, pulse_width, phases, phase_reset, interp_data)

    def pdq_pulse(self, N, dt, stack_amp, stack_assign, data_waveform, special, init_trigger):
        stepwidth = dt/self.dac_divider
        stepsize = int(stepwidth)
        p = float(stepsize)/stepwidth
        #print('N = %i, T = %.4f us (%i) dt = %.4f us (%i), %.2f%%' % (N,N*dt*1e6,stepsize*N,dt*1e6,stepwidth,p*100))
        if stepwidth<1:
            print(bcolors.FAIL + ('Step size too small: %f (minimum 1)!' % (stepwidth)) + bcolors.ENDC)
            return
        elif (p<1.):
            print(bcolors.FAIL + ('Sampling does not match pdq sampling: %.2f%% deviation!' % ((1-p)*100) ) + bcolors.ENDC)
            return
        
        #last = False
        for step in range(N):            
            if step == 0:           # first
                #init_trigger = {False: one trigger required (osci mode), True: two trigger (eios mode, pdq_init)}
                segment = {"trigger": init_trigger, "duration":stepsize, "wait":False}
            elif step == (N-1):     # last
                #print('LAST',step,N,init_trigger,self._device)
                #last = True
                segment = {"trigger": False, "duration":stepsize, "wait":True}
            else:                   # intermediate
                segment = {"trigger": False, "duration":stepsize, "wait":False}
            channel_data = []
            for c in range(self._numchannels):
                #cv = stack_amp[c]
                lv = self._last_voltages[c]
                ix = stack_assign[c]
                if ix < 0:
                    channel_data.append({"bias": {"amplitude":[lv]}})
                else:
                    data = data_waveform[ix]
                    spline = data[step]
                    '''
                    for i in range(len(spline)):
                        spline[i] *= cv
                        if last and i>0:
                            spline[i] = 0.
                    '''
                    channel_data.append({"bias": {"amplitude":spline}})
            segment['channel_data'] = channel_data
            self._program.append(segment)
        #self._pdq_const(self._last_voltages, False)
        #print(json.dumps(self._program, indent = 4))
        
    def _pdq_set_dds(self, steps, amplitudes, frequencies, phases, phase_reset=0, trigger=True):
        segment = {'trigger': trigger, 'duration':steps, 'channel_data':list()}
        only_nonzero_channels = (phase_reset>1)
        bphase_reset = ((phase_reset==1) or (phase_reset==3))
        for c in range(self._numchannels):
            amp = amplitudes[c]
            phase = phase_modulo(phases[c])
            freq = frequencies[c]/self.freq
            if only_nonzero_channels and (abs(amp) <= 1e-5):
                single_channel_data={"bias": {"amplitude":[self._last_voltages[c]]}}
            else:
                single_channel_data={
                    "dds":{
                    "amplitude":[amp, 0., 0., 0.],
                    "phase": [phase, freq],   # phase element of [0, 0.5[
                    "clear": bphase_reset
                    }}
            segment['channel_data'].append(single_channel_data)
        self._program.append(segment)
        self._last_frequencies = frequencies
        self._last_phases = phases

    def _pdq_dds_pulse(self, amplitudes, frequencies, pulse_width, phases, phase_reset, interp_data):
        only_nonzero_channels = (phase_reset>1)
        if (pulse_width<=0.02):
            pulse_width=1.
            print("pulse_width is too small!! setting to 1")
        amplitude_scaling = self.dac_divider*1e6/pulse_width
        power_list = [1., amplitude_scaling, amplitude_scaling**2, amplitude_scaling**3]
        pdq_freq = self.freq*1e-6 # in MHz
        steps = pdq_freq*pulse_width
        for i in range(len(interp_data)):
            idata, step_duration = interp_data[i]
            steps_per_interp = int(step_duration*steps)
            trigger = i==0
            if steps_per_interp <= 1:
                steps_per_interp=1
            segment = {"trigger": trigger, "duration":steps_per_interp}
            channel_data=list()
            bphase_reset=(phase_reset>0) and (i==0)
            for c in range(self._numchannels):
                amp = amplitudes[c]
                phase = phase_modulo(phases[c])
                freq = frequencies[c]/self.freq
                if only_nonzero_channels and (abs(amp) <= 1e-5):
                    single_channel_data={"bias": {"amplitude":[self._last_voltages[c]]}}
                else:
                    single_channel_data={
                        "dds":{
                            "amplitude":[
                                    amp*idata[0]*power_list[0],
                                    amp*idata[1]*power_list[1],
                                    amp*idata[2]*power_list[2],
                                    amp*idata[3]*power_list[3]
                            ],
                            "phase": [phase, freq],   # phase element of [0, 0.5[
                            "clear": bphase_reset
                        }
                    }
                channel_data.append(single_channel_data)
            segment['channel_data'] = channel_data
            self._program.append(segment)
        self._last_frequencies = frequencies
        self._last_phases = phases

    def _pdq_chirp_start(self, amplitudes, start_freq, stop_freq, sweep_time, phases, points, phase_reset=False):
        points = int(points)
        stepsize = int( float(sweep_time/self.dac_divider) / float(points) )
        steps = int(points*stepsize)
        #print(points,steps,stepsize)

        for step in range(0, steps, stepsize):
            # first step: wait for trigger
            segment = {"trigger": (step == 0), "duration": stepsize, "wait":False}
            if (steps-stepsize == 0):
                p = 1
            else:
                p = (float(step)/float(steps-stepsize))
            freq = start_freq + (stop_freq-start_freq)*p
            freq_pdq = freq/self.freq
            #total_phase = sweep_time*freq
            #phi = total_phase*p
            phi = 0
            channel_data = list()
            for c in range(self._numchannels):
                phase = phase_modulo(phases[c])
                amp = amplitudes[c]
                single_channel_data={
                    "dds":{
                    "amplitude":[amp, 0., 0., 0.],
                    "phase": [phase, freq_pdq],
                    "clear": phase_reset
                    }}
                channel_data.append(single_channel_data)
            #print(p*100,amp,phase,freq,freq_pdq,total_phase,phi)
            segment['channel_data'] = channel_data
            self._program.append(segment)

    def _pdq_chirp_stop(self):
        self._pdq_set_dds(self._ttl_steps, self._zero_list, self._last_frequencies, self._last_phases, phase_reset=False, trigger=False)

    def pdq_write(self):
        with Serial(self._device) as serial_dev:
            dev = Pdq2(dev=serial_dev)
            dev.cmd("START", False)
            dev.cmd("DCM", self.multiplier)      #run @ 100MHz
            #dev.cmd("ARM", False)
            dev.cmd("TRIGGER", False)
            if self.reset:
                dev.write(b"\x00\x00")  # flush any escape
                dev.cmd("RESET", True)
                #time.sleep(.1)
            dev.program([self._program])
            dev.cmd("START", True)
            dev.cmd("ARM", True)
            #dev.close()
            del dev
            #print(sys.getrefcount(serial_dev))
        #print(json.dumps(self._program, indent = 4))

#################### Step functions for bias mode ###########################

def chirp(f0,f1,ts,phi,t,dac_divider):
    m = (f1-f0)/ts
    f = f0+m*t
    x0 = 2.*pi*f*t+phi
    x1 = 2.*pi*(f0 + 2.*m*t)
    x2 = 2.+pi*2.*m
    y0 = sin(x0)
    y1 = (cos(x0)*x1)*dac_divider
    y2 = (-sin(x0)*x1**2 + cos(x0)*x2)*dac_divider**2
    y3 = (-cos(x0)*x1**3 - 3.*sin(x0)*x1*x2)*dac_divider**3
    return y0, y1, y2, y3

def linear_step(lvoltage, dvoltage, steps):
    # linear
    a0 = lvoltage
    a1 = dvoltage/steps
    return a0, a1

def smooth_step(lvoltage, dvoltage, steps):
    #cubic smooth
    #following from comparison of coefficients in taylor expansion
    #and https://en.wikipedia.org/wiki/Smoothstep f(x) = 3x^2 -2x^3
    a0 = lvoltage
    a1 = 0.
    a2 = 6./(steps)**2. * dvoltage
    a3 = -12./(steps)**3. * dvoltage
    return a0, a1, a2, a3


def overshoot1(lvoltage, dvoltage, tstep):
    #overshoot through filter
    R = 9.3e3
    C = 10.0e-9
    a1 = dvoltage / tstep
    a0 = lvoltage + (R*C*a1)
    return a0, a1

def overshoot3(lvoltage, dvoltage, steps, dac_divider):
    # cubic smooth through first-order lowpass filter, analytically derived
    # target function: f(x) = - 2* x**3 + 3 * x**2
    # overshooting function: g(x) = f(x) + R * C * f'(x)
    R = 0.470e3 # Ohm
    C = 2.2e-9 # Farad
    tau = R * C / dac_divider # in units of clock!
    print("lowpass frequency:", round(1/(2*3.141592*R*C)), "Hz")
    a0 = lvoltage
    a1 = 6. / steps**2 * dvoltage * tau
    a2 = 6. / steps**2 * dvoltage  - 12. / steps**3 * dvoltage * tau
    a3 = -12. / steps**3 * dvoltage
    return a0, a1, a2, a3

def overshoot4(lvoltage, dvoltage, steps, dac_divider):
    # cubic smooth through second-order lowpass filter, analytically derived
    # target function: f(x) = - 2* x**3 + 3 * x**2
    # overshooting function 1st order: g(x) = f(x) + R0 * C0 * f'(x)
    # overshooting function 2nd order: h(x) = g(x) + R1 * C1 * g'(x) + (g(x) - f(x)) * R1 / R0

    #print("lowpass frequencies:", round(1/(2*3.141592*R0*C0)), "and", round(1/(2*3.141592*R1*C1)), "Hz")

    R0 = 1.18e3 # Ohm
    C0 = 4.84e-9 # Farad
    R1 = 50 # Ohm
    C1 = 2.89e-9 # Farad
    tau0 = R0 * C0 / dac_divider # in units of clock!
    tau1 = R1 * C1 / dac_divider # in units of clock!
    a0 = lvoltage + 6 * dvoltage * tau0 * tau1 / (steps**2)
    a1 = 6. / steps**2 * dvoltage * tau0 + 6 * dvoltage * tau0 * R1 / (R0 * steps**2) + (6 * dvoltage / (steps**2) - 12 * dvoltage * tau0 / (steps**3) ) * tau1
    a2 = 6. / steps**2 * dvoltage  - 12. / steps**3 * dvoltage * tau0 - 12 * dvoltage * R1 * tau0 / (R0 * steps**3 ) - 12 * dvoltage * tau1 / (steps**3)
    a3 = -12. / steps**3 * dvoltage
    return a0, a1, a2, a3

def overshoot21(lv, dv, step, steps, tdelay, C0, R0):
    # smooth through three first-order lowpass filters, analytically derived, max order of smoothstep: t**13
    x = step-0.1*tdelay
    f = C0*R0*dv
    a0 = ( lv + f*((12012*x**12)/steps**13 - (72072*x**11)/steps**12 + 
           (180180*x**10)/steps**11 - (240240*x**9)/steps**10 + (180180*x**8)/steps**9 - 
           (72072*x**7)/steps**8 + (12012*x**6)/ steps**7) + dv*((924*x**13)/steps**13 - 
           (6006*x**12)/steps**12 + (16380*x**11)/steps**11 - (24024*x**10)/steps**10 + 
           (20020*x**9)/steps**9 - (9009*x**8)/steps**8 + (1716*x**7)/steps**7) )
    # a0 = lv + dv * (step/steps)
    a1 = ( f*( (144144*x**11)/steps**13 - (792792*x**10)/steps**12 + 
           (1801800*x**9)/steps**11 - (2162160*x**8)/steps**10 + (1441440*x**7)/steps**9 - 
           (504504*x**6)/steps**8 + (72072*x**5)/steps**7) + dv*((12012*x**12)/steps**13 - 
           (72072*x**11)/steps**12 + (180180*x**10)/steps**11 - (240240*x**9)/steps**10 + 
           (180180*x**8)/steps**9 - (72072*x**7)/steps**8 + (12012*x**6)/steps**7 ) )
    # a1 = 0.
    a2 = ( f*( (1585584*x**10)/steps**13 - (7927920*x**9)/steps**12 + 
           (16216200*x**8)/steps**11 - (17297280*x**7)/steps**10 + (10090080*x**6)/steps**9 - 
           (3027024*x**5)/steps**8 + (360360*x**4)/steps**7) + dv*((144144*x**11)/steps**13 - 
           (792792*x**10)/steps**12 + (1801800*x**9)/steps**11 - (2162160*x**8)/steps**10 + 
           (1441440*x**7)/steps**9 - (504504*x**6)/steps**8 + (72072*x**5)/steps**7) )
    # a2 = 0.
    a3 = ( f*( (15855840*x**9)/steps**13 - (71351280*x**8)/steps**12 + 
           (129729600*x**7)/steps**11 - (121080960*x**6)/steps**10 + (60540480*x**5)/steps**9 - 
           (15135120*x**4)/steps**8 + (1441440*x**3)/steps**7) + dv*((1585584*x**10)/steps**13 - 
           (7927920*x**9)/steps**12 + (16216200*x**8)/steps**11 - (17297280*x**7)/steps**10 + 
           (10090080*x**6)/steps**9 - (3027024*x**5)/steps**8 + (360360*x**4)/steps**7) )
    # a3 = 0.
    return a0, a1, a2, a3

interp_data_gaus=[
    [ [-0.00051739819886974941, 0.023742545916308498, -0.19714810725293302, 1.1488543637573951], 0.8],
    [ [0.061259979812610235, 	0.15426684897008419, 	2.2042133638783978, 	-1.9217072484437228], 		0.8],
    [ [0.72090086643653628, 	1.3758361707524562, 	-3.4395904268811286, 	-5.2605577846418163e-14], 	0.8],
    [ [0.72603605022923501, 	-1.3026912205708157, 	0.66684756512343824, 	1.9217072484436746], 		0.8],
    [ [0.053424816587202974, 	-0.23365745651632924, 	0.72193538375298738, 	-1.148543637574069], 		0.8],
    [ [0., 0., 0., 0.], 0]]

interp_data_rect=[
    [[1.,0., 0., 0.],1],
    [[0.,0., 0., 0.],0]]

interp_data_sinc=[
    [ [-0.0036469484008280046, 0.13211979706941956, -0.34815308596634431, 0.28415009608904124], 1.368421052631579],
    [ [-0.024296962508383188, -0.055658384094331728, 0.43753458666139639, -0.67150809610036355], 1.368421052631579],
    [ [0.026654557399724089, -0.11678782166204758, 0.039440599010161483, 0.26093294752437279], 1.368421052631579],
    [ [0.0070113526098990967, 0.1836142545253435, -0.61792891675505712, 0.64441361348085124], 1.368421052631579],
    [ [-0.042780867670076952, -0.018834100970206571, 0.55578041819417789, -1.0054783840037997], 1.368421052631579],
    [ [0.031882933858115178, -0.2485950902179582, 0.33673365256539461, 0.095372798227800384], 1.368421052631579],
    [ [0.033346459789872919, 0.30514394175740894, -1.2911734801340289, 1.5527055312067508], 1.368421052631579],
    [ [-0.092763775850771946, 0.063669175438306663, 1.0982288887515115, -2.3490242352954165], 1.368421052631579],
    [ [0.042967555015707218, -0.74268170366692465, 0.76724772916436657, 1.5458337590394049], 1.368421052631579],
    [ [0.36764691484126027, 1.8089290675799043, -2.6438194064629363, 1.0873410202333558e-15], 1.368421052631579],
    [ [0.40522397785021907, -1.7545819771156681, 2.8825991889024905, -1.5458337590393922], 1.368421052631579],
    [ [0.019402693163288656, 0.63285470402596944, -2.1162253279685252, 2.3490242352954072], 1.368421052631579],
    [ [-0.094870916310246245, 0.007945337019123519, 0.83358145730679234, -1.5527055312067575], 1.368421052631579],
    [ [0.047711584817610883, -0.30149474404326493, 0.46724379750869921, -0.095372798227798178], 1.368421052631579],
    [ [0.022398258693559797, 0.19970991040333827, -0.82013737044260049, 1.0054783840037991], 1.368421052631579],
    [ [-0.04507114221483674, 0.058615893730807181, 0.26390023853453065, -0.64441361348085391], 1.368421052631579],
    [ [0.015206330788168821, -0.18149193505334746, 0.39650673772772177, -0.26093294752436952], 1.368421052631579],
    [ [0.022410498188524874, 0.085651876258301105, -0.48137122905488988, 0.67150809610036288], 1.368421052631579],
    [ [-0.02746854843564257, 0.078253864950741145, 0.04068388762918778, -0.28415009608904412], 1.368421052631579],
    [[0.,0., 0., 0.],0]]


