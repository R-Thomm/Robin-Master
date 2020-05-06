from datetime import datetime
import numpy as np
import glob
import matplotlib.pyplot as plt
from . import eios_data as eios_file

def extract_xml(fn):
    _, xml = eios_file.read(fn, sort=False, skip_first=False)
    xml_dict = eios_file.read_xml(xml)
    return xml_dict

def shim_parser(xml_dict, shim_name):
    #for i, (key, value) in enumerate(shims.items()):
    #   pass
    shims = xml_dict['shims']
    sh_found = int(shim_name in shims.keys())
    return sh_found

def profile_parser(xml_dict, profile_name, shim_names):
    shim_list = []
    profs = xml_dict['profiles'][profile_name]
    for sh in shim_names:
        if sh in profs.keys():
            shim_list.append(profs[sh])
        else:
            shim_list.append(None)
    return shim_list

def parse(parser_func, path='/home/qsim/Results/tiamo4.sync', year='*', month='*', day='*', cat='*', scr='*', name='*.dat'):
    file_list = glob.glob(path+'/'+year+'/'+month+'/'+day+'/'+cat+'/'+scr+'/'+name, recursive=True)
    N = len(file_list)
    print('N = %i'%N)
    #file_list = file_list[:100]
    time_list = []
    tstr_list = []
    data_list = []
    
    n = N//10
    for i,fn in enumerate(file_list):
        if i%n==0 or i==N-1:
            print('%6.2f %%'%((i/(len(file_list)-1))*100))

        xml_dict = extract_xml(fn)
        # {('time', 'h'): 14, ('time', 'm'): 44, ('time', 's'): 30, ('date', 'd'): 12, ('date', 'm'): 10, ('date', 'y'): 2018}
        t_dct = xml_dict['timestamp']
        t_str = '%04i-%02i-%02i %02i:%02i:%02i'% (t_dct['date', 'y'], t_dct['date', 'm'], t_dct['date', 'd'], t_dct['time', 'h'], t_dct['time', 'm'], t_dct['time', 's'])
        t_stp = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
        t = t_stp.timestamp()
        time_list.append(t)
        tstr_list.append(t_str)

        data_list.append(parser_func(xml_dict))

    tstr_list = np.array(tstr_list)
    time_list = np.array(time_list)
    data_list = np.array(data_list)

    ind = np.argsort(time_list)
    time_list = time_list[ind]
    tstr_list = tstr_list[ind]
    data_list = data_list[ind]

    data = { 'timestring':tstr_list.tolist(), 'timestamp':time_list.tolist(), 'data':data_list.tolist() }

    return data

if __name__ == "__main__":
    parser = lambda x: profile_parser(x,profile_name='EU_cool',shim_names=['ExExpZ'])
    data = parse(parser,year='2019', month='06_Juni', day='1*', cat='ion_shuttle', scr='low_RF_Exp_Load_Exp')
    
    time_list = data['timestamp']
    data_list = data['data']

    #import json
    #with open('scan_Ey_10Vpm.json', "w") as write_file:
    #    json.dump(data, write_file, indent=4)

    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.plot(time_list,data_list,'x--')
    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i,l in enumerate(labels):
        timestamp = float(l)
        labels[i] = datetime.utcfromtimestamp(timestamp).strftime('%d.%m.%y')

    #ax.set_xticklabels(labels,rotation='vertical')
    ax.set_xticklabels(labels)
    plt.show()

