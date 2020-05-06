import requests
import matplotlib.pyplot as plt
import numpy as np

#print(ret.content)
#print(dir(ret))

class http_api(object):
    def __init__(self, host='127.0.0.1', port=8000, protocol='http'):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.baseurl = '%s://%s:%i'%(self.protocol, self.host, self.port)
        
    def get(self, path, data):
        url = '%s/%s'%(self.baseurl, path)
        return requests.get(url,json=data)
        
    def post(self, path, data):
        url = '%s/%s'%(self.baseurl, path)
        return requests.post(url,json=data)
        
    def parse(self, ret):
        status = ret.status_code
        mimetype = ret.headers["Content-Type"]
        data = {}
        if status==200:
            if mimetype == 'application/json':
                data = ret.json()
            else:
                data = ret.text
        else:
            pass
        return status, data
    
class web_lock_client(http_api):
    def get(self, cmd, data):
        ret, msg = super().parse(super().get('get/%s'%cmd, data))
        if ret==200:
            stat = msg['status']
            data = msg['data']
            return stat, data
        else:
            return False, {}

    def post(self, cmd, data):
        ret, msg = super().parse(super().post('post/%s'%cmd, data))
        if ret==200:
            stat = msg['status']
            data = msg['data']
            return stat, data
        else:
            return False, {}
    
    def index(self):
         return self.get('', {})
    
    def stop(self):
        return self.post('stop', {})

    def get_list(self):
        return self.get('list', {})
        
    def get_config(self):
        return self.get('config', {})
    
    def get_default_config(self):
        return self.get('config/default', {})
    
    def get_status(self, name = None):
        return self.get('status', {'name':name})
    
    def get_parameter(self, name):
        return self.get('parameter', {'name':name})
    
    def get_trace_last(self, name):
        return self.get('trace/last', {'name':name})
        
    def get_trace(self, name):
        return self.get('trace', {'name':name})
        
    def get_graph(self, name):
        return self.get('graph', {'name':name})
    
    def set_trace(self, name, tracelen):
        return self.post('trace', {'name':name, 'data':tracelen})
    
    def clear_trace(self, name):
        return self.post('trace/clear', {'name':name})
        
    def set_offset(self, name, offset):
        return self.post('offset', {'name':name, 'data':offset})
        
    def set_sampling(self, sampling):
        return self.post('sampling', {'data':sampling})
    
    def set_setpoint(self, name, setpoint):
        return self.post('setpoint', {'name':name, 'data':setpoint})

    def set_pid(self, name, P, I, D):
        par_dict = {'P':P, 'I':I, 'D':D}
        return self.set_pid(name, par_dict)
    
    def set_pid_dict(self, name, pid_dict):
        return self.post('pid', {'name':name, 'data':pid_dict})
    
    def set_limit(self, name, lim_low, lim_high):
        return self.post('limits', {'name':name, 'data':[lim_low,lim_high]})
    
    def set_range(self, name, center, span):
        return self.post('range', {'name':name, 'data':[center, span]})

    def activate(self, name):
        return self.post('activate', {'name':name})
    
    def deactivate(self, name):
        return self.post('deactivate', {'name':name})
    
    def lock(self, name):
        return self.post('lock', {'name':name})
    
    def unlock(self, name):
        return self.post('unlock', {'name':name})
    
    def remove(self, name):
        return self.post('remove', {'name':name})
    
    def reset(self, name):
        return self.post('reset', {'name':name})
