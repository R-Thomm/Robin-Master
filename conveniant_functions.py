import numpy as np
import matplotlib.pyplot as plt
from time import localtime, strftime



def select_files(files,start,end,print_len=True):
    res = []
    for file in files:
        h = int(file[-23:-21])
        m = int(file[-20:-18])
        s = int(file[-17:-15])
        if start[0]*3600+start[1]*60+start[2] <= h*3600+m*60+s <= end[0]*3600+end[1]*60+end[2]:
            res.append(file)
    if print_len:
        print(len(res),'files')
    return res

def save_arr(array, note=''):
    """saves a numpy array in a file in the directory 'data/', named after the time of saving, with note (string) as suffix"""
    time_str = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    if note == '':
        path = 'data/' + time_str
    else:
        path = 'data/' + time_str + '_' + note
    
    rr = np.savez(path,array)
    print(path)
    return rr


from PyModules.analyse_eios.eios_data import read, read_xml
def get_ionprop(file, ion_props = ['blue_scale', 'dec_lf', 'fr_lf'], verbose=True):
    _,xml = read(file)
    xml_dict = read_xml(xml)
    # print(xml_dict)
    vals = []
    for ion_prop in ion_props:
        value = xml_dict['ionproperties'][ion_prop]
        if verbose:
            print(ion_prop,'=',value)
        vals.append(value)
    return vals


def gauss(x, A, C, mu, sig):
    return(A/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*(x-mu)**2/sig**2) + C)


def W_from_fock_simple(focks):
    w = 0
    for i, f in enumerate(focks):
        w += (-1)**i * f
        return 2/np.pi * w
