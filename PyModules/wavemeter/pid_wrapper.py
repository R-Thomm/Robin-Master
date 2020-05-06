#https://pypi.org/project/simple-pid/
from simple_pid import PID
import collections
import time

class logger(object):
    def __init__(self, tracelen):
        self.trace = []
        self.setup(tracelen)
        
    def setup(self, tracelen):
        self.trace = collections.deque(self.trace, maxlen=tracelen)
        
    def get(self):
        return list(self.trace)
    
    def get_item(self, index):
        return self.trace[index]
    
    def append(self, val):
        self.trace.append(val)
        
    def clear(self):
        self.trace.clear()
        

class pid_container(object):
    def __init__(self, 
                func_read, func_write, 
                lock, offset, 
                P, I, D, setpoint, limits, 
                tracelen):
        self.last_out = None
        self.lock = lock
        self.offset = offset
        self.pid = PID(Kp=P, Ki=I, Kd=D, 
                       setpoint=setpoint, sample_time=None,
                       output_limits=tuple(i for i in limits),
                       auto_mode=True,
                       proportional_on_measurement=False)
        
        self.func_write = func_write
        self.func_read = func_read

        self.times = logger(tracelen)
        self.trace = logger(tracelen)
        self.error = logger(tracelen)
        self.outpt = logger(tracelen)
    
    def reset(self):
        self.pid._integral = 0
        
    def clear_trace(self):
        self.times.clear()
        self.trace.clear()
        self.error.clear()
        self.outpt.clear()
        
    def set_trace(self, tracelen):
        self.times.setup(tracelen)
        self.trace.setup(tracelen)
        self.error.setup(tracelen)
        self.outpt.setup(tracelen)
        
    def set_offset(self, offset):
        self.offset = offset
        
    def set_setpoint(self, sp):
        self.pid.setpoint = sp
        
    def set_limits(self, limits):
        self.pid.output_limits = tuple(i for i in limits)
        
    def set_pid(self, kp, ki, kd):
        self.pid.tunings = (kp,ki,kd)
    
    def get_trace(self):
        return self.times.get(), self.trace.get(), self.error.get(), self.outpt.get()
        
    def get_trace_last(self):
        t = self.times.get_item(-1)
        y = self.trace.get_item(-1)
        e = self.error.get_item(-1)
        o = self.outpt.get_item(-1)
        return t,y,e,o
        
    def set_lock(self, state):
        self.lock = state
    
    def __measure(self):
        val = self.func_read()
        now = time.time()
        return now, val
    
    def __output(self, out):
        value = self.offset + out
        if callable(self.func_write):
            self.last_out = self.func_write(value, self.last_out)
    
    def __call__(self):
        now, value_in = self.__measure()
        if self.lock:
            value_pid = self.pid(value_in)
            self.__output(value_pid)
        value_out = self.last_out
        self.times.append(now)
        self.trace.append(value_in)
        self.error.append(value_in-self.pid.setpoint)
        self.outpt.append(value_out)
        return now, value_in, value_out
    
    

