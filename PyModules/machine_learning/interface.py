import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from threading import Thread

import sys
sys.path.insert(0, './../')

from PyModules.pdq.pdq_waveform import PDQ_Waveform
from PyModules.osci.osci import Oscilloscope
from PyModules.eios import trigger

class Interface(object):
    def __init__(self, serial_numbers=["ABCDEFG0"], pdq_ch_per_stck = 3, multiplier = True):
        self.pdq_wave = PDQ_Waveform(serial_numbers=serial_numbers, pdq_ch_per_stck=pdq_ch_per_stck, multiplier=multiplier)  
        self.osci = Oscilloscope()
        usb = self.osci.list_device()
        self.adr = usb[-1]
        for name in usb:
            if (name.find('DS1Z')>-1):
                self.adr = name
                break
        print('Oscilloscope Address\n\t%s'%self.adr)
        self.osci.open(self.adr)

        self.trg = trigger()
        
    def __del__(self):
        del self.osci
        del self.pdq_wave
        
    def read(self,channel):
        t, data, dt = self.osci.read(channel)
        if data.shape[0]<len(channel):
            raise Exception('Not enough channels received!')
        #T = np.linspace(0., (len(t)-1)*dt,  num=len(t))
        #return T,data
        return t,data

    def measure(self, times, data_in, data_init, shim, ch_in, interp_order=0):
        # PDQ
        self.pdq_wave.send(times,shim,data_in,data_init,order=interp_order)

        # Oscilloscope

        # start trigger w/ delay 250 ms
        thread = Thread(target = self.trg.run, args = (.25,))
        thread.start()

        # arm osci
        t,data_out = self.read(ch_in)

        # stop trigger loop
        self.trg.stop()
        thread.join()

        return t,data_out

