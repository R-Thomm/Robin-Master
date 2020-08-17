import numpy as np
import matplotlib.pyplot as plt



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

def save_arr(array, note):
    """saves a numpy array in a file in the directory '/data/', named after the time of saving, with note (string) as suffix"""


from PyModules.analyse_eios.eios_data import read, read_xml
def get_ionprop(file, ion_props = ['blue_scale', 'dec_lf', 'fr_lf'], verbose=True):
    _,xml = read(file)
    xml_dict = read_xml(xml)
    vals = []
    for ion_prop in ion_props:
        value = xml_dict['ionproperties'][ion_prop]
        if verbose:
            print(ion_prop,'=',value)
        vals.append(value)
    return vals
