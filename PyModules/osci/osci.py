#!/usr/bin/env python

"""
Current version by Jan-Philipp Schoeder

Major improvements: Yanis Taege and Fritz Bayer

Inspired by pklaus/rigol-plot.py, forked from shirriff/rigol-plot.py
https://gist.github.com/pklaus/7e4cbac1009b668eafab

Download data from a Rigol DS1052E oscilloscope and graph with matplotlib.
By Ken Shirriff, http://righto.com/rigol

Based on http://www.cibomahto.com/2010/04/controlling-a-rigol-oscilloscope-using-linux-and-python/
by Cibo Mahto.
"""

import numpy
import visa
import os
import time as tme

class ResourceManager(object):
    def __init__(self, **kwargs):
        if os.name == 'posix':
            self.rm = visa.ResourceManager('@py',**kwargs)
            self.rm.write_termination='\n'
            self.rm.read_termination = '\n'
        elif os.name == 'nt':
            self.rm = visa.ResourceManager('@ni',**kwargs)
        else:
            self.rm = visa.ResourceManager(**kwargs)
    def __enter__(self):
        return(self.rm)

    def __exit__(self, exc_type, exc, exc_tb):
        del self.rm

class Instrument(object):
    def __init__(self, rm, resource_name, **kwargs):
        self.inst = rm.open_resource(resource_name, **kwargs)
        self.inst.timeout = 25000
        self.query_delay = self.inst.query_delay
        
    def __enter__(self):
        #return(self.inst)
        return(self)
        
    def __exit__(self, exc_type, exc, exc_tb):
        self.inst.close()
        del self.inst
    
    def query(self,message):
        self.write(message) #Request the data
        rawdata = self.read_raw() #Read the block of data
        data = rawdata[:rawdata.find(b'\n')].decode('utf-8')
        return(data)
    
    def query_raw(self,message):
        self.write(message) #Request the data
        rawdata = self.read_raw() #Read the block of data
        head = rawdata[:11]
        data = rawdata[11:-1] #Drop the heading
        return head,data
    
    def read_raw(self):
        return(self.inst.read_raw()) #Read the block of data
    
    def write(self,message):
        ret = self.inst.write(message)
        #if not ret[1]==0:
        #    err = self.inst.query('SYST:ERR?')
        #    print('WRITE: "%s"' % message, ret, err)
        return(ret)

class Oscilloscope(object):
    def __init__(self, verbose=False, fast=False, **kwargs):
        self.verbose = verbose
        self.fast = fast
        self.channels = [1]
        self.first = True

        rm = ResourceManager(**kwargs)
        self.rm = rm.rm
        self.scope = []
    
    def __del__(self):
        if self.scope:
            self.scope.inst.close()
            del self.scope
        del self.rm
        
    def list_device(self):
        instruments = self.rm.list_resources(query=u'?*::INSTR')
        usb = list(filter(lambda x: 'USB' in x, instruments))
        if len(usb)<1:
            raise Exception('No Oscilloscope found!')
        return usb
    
    def open(self,adr):
        self.adr = adr
        self.scope = Instrument(self.rm, self.adr)

        self.name = self.scope.query('*IDN?')
        
        #scope.write(':ACQuire:MDEPth 600000')
        #scope.write(':CHANnel'+str(channel)+':COUPling '+ch_cpl_mode)
        
        if self.verbose:
            print(self.name)
            mem_depth = self.scope.query(':ACQuire:MDEPth?')
            sample_rate = self.scope.query(':ACQuire:SRATe?')
            print('Sampl.rate: %s; Mem.depth: %s' % (sample_rate,mem_depth))    
    
    def read(self,channels=None):
        if channels is None:
            channels = self.channels
        else:
            self.first = True
            self.channels = channels
            
        self.scope.write(':SINGle')

        # Get time scale and offset
        if (not self.fast) or self.first:
            self.timescale = float(self.scope.query(':TIM:SCAL?'))
            self.timeoffset = float(self.scope.query(':TIM:OFFS?'))

        self.scope.write(':WAVeform:FORMat BYTE')
        self.scope.write(':WAVeform:MODE NORMAL')

        trigger = 'WAIT'
        while trigger.find('STOP') < 0:
            trigger = self.scope.query(":TRIGger:STATus?")
            tme.sleep(0.001)

        # Record data of a single trigger
        ch_data = []
        
        if self.first:
            self.y_inc = [0.]*len(channels)
            self.y_ref = [0.]*len(channels)
            self.y_ori = [0.]*len(channels)
        for i,channel in enumerate(channels):
            self.scope.write(':WAV:SOUR CHAN'+str(channel))
            
            # Get y-axis parameter
            if (not self.fast) or self.first:
                self.y_inc[i] = float(self.scope.query(':WAVeform:YINCrement?'))
                self.y_ref[i] = float(self.scope.query(':WAVeform:YREFerence?'))
                self.y_ori[i] = float(self.scope.query(':WAVeform:YORigin?'))

            _,rawdata = self.scope.query_raw(':WAV:DATA? CHAN'+str(channel))
            data = numpy.frombuffer(rawdata, 'B').astype(float)
            data = (data - self.y_ori[i] - self.y_ref[i]) * self.y_inc[i]
            data_size = len(data)
            ch_data.append(data)

            if self.verbose:
                cpl = self.scope.query(':CHANnel'+str(channel)+':COUPling?')
                print('CH%d(%s): data size: %d' % (channel,cpl,data_size))
                
        self.scope.write(':RUN')
        
        # Generate a time axis
        if data_size>0.:
            dt = self.timescale*12./float(data_size)
        else:
            dt = 0.
        time = numpy.linspace(self.timeoffset - 6 * self.timescale, 
                              self.timeoffset + 6 * self.timescale, 
                              num=data_size)
        
        ch_data = numpy.array(ch_data)
        self.first = False
        return time, ch_data, dt


