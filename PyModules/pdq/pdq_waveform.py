#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import json

from . import eios_pdq2 as EP2
from .waveform import func_zeros

def load_json_file(data_file):
    with open(data_file, "r") as read_file:
        data = json.load(read_file)
    return data

def save_json_file(data_file,data):
    with open(data_file, "w") as write_file:
        json.dump(data, write_file, indent=4)

def save_wvf(times, data_wf, data_file):
    data = dict()
    data['times'] = times.tolist()
    data['data'] = data_wf.tolist()
    save_json_file(data_file, data)

class PDQ_Waveform(object):
    def __init__(self, serial_numbers, pdq_ch_per_stck, multiplier):
        self.serial_numbers = serial_numbers
        self.pdq_ch_per_stck = pdq_ch_per_stck
        self.multiplier = multiplier
        self.pdq_count = len(serial_numbers)
        self.f_pdq = 50e6
        if self.multiplier:
            self.f_pdq *= 2.
        self.dac_divider = 1./self.f_pdq

        self.fs = self.f_pdq
        self.dt = self.dac_divider

        self.numchannels = self.pdq_ch_per_stck*self.pdq_count
        
        print("#PDQs = %d" % self.pdq_count)
        print('PDQ min. time step: %e' % self.dt)
    
    def convert(self, data, upper = 9.9999, lower = -9.9999):
        wave_data = []
        for ch_data in data:
            # clamp data to valid pdq voltages
            ch_data[0,ch_data[0,]>upper] = upper
            ch_data[0,ch_data[0,]<lower] = lower
            wave_data.append(ch_data.tolist())
        return wave_data

    def send(self, times, shim, data, voltage_init, order=0,special=0,init_trigger=False):
        # sampling times
        times = times.tolist()
        # selected channel
        shim = shim.tolist()
        # initialization voltages
        voltage = voltage_init.tolist()
        
        # Convert data to list and respect limits
        wave_data = self.convert(data)
        #print(len(wave_data), len(wave_data[0]), len(wave_data[0][0]))
        thisPDQ = EP2.EIOS_PDQ2(self.serial_numbers, voltage, self.multiplier, self.pdq_ch_per_stck)
        thisPDQ.pdq_pulse(times,shim,wave_data,order=order,special=special, init_trigger=init_trigger)
        thisPDQ.pdq_write()
        del thisPDQ
    
    def save(self, times, data_wf, data_file = './waveform.json'):
        save_wvf(times, data_wf, data_file)

    def send_file(self, shim, voltage_init, data_file = './waveform.json'):
        data = load_json_file(data_file)
        times = np.array(data['times'])

        wave_data = np.array(data['data'])
        #print(len(data['data']))
        #print(wave_data.shape)
        self.send(times, shim, wave_data, voltage_init)

