import numpy as np
from lxml import etree
import glob

def read(filename, sort=False, skip_first=True):
    f=open(filename, 'br')
    content = f.read().splitlines()
    f.close()
    xml = list()
    i=0
    while(i<len(content) and len(content[i])>0 and content[i][0]==35): # '#'
        xml.append(content[i][1:])
        i+=1
    if not (xml[-1]==b'</metadata>'):
        xml.append(b'</metadata>')
    xml_str = b'\n'.join(xml)
    root = etree.fromstring(xml_str)

    data = []
    while(i<len(content)):
        x=[]
        y=[]
        y_err=[]
        t=[]
        strt = i
        while(i<len(content) and content[i]==b''):
            i+=1
        if (not i<len(content)):
            break
        while(i<len(content) and content[i]!=b''):
            columns = content[i].split(bytes([9])) # '\t'
            x.append(columns[0])
            y.append(columns[1])
            y_err.append(columns[2])
            if len(columns)>3:
                t.append(columns[3])
            else:
                t.append(0)
            i+=1
        x=np.array(x, dtype='float64')
        y=np.array(y, dtype='float64')
        y_err=np.array(y_err, dtype='float64')
        t=np.array(t, dtype='float64')

        hists=list()
        for j in range(len(x)):
            chist=list()
            while(content[i]==b''):
                i+=1
            while(content[i]!=b''):
                columns = content[i].split(bytes([9])) # '\t'
                cbin=np.int(columns[0])
                ccount=np.int(columns[1])

                for c in range(ccount):
                    chist.append(cbin)
                i+=1
            #print(chist)
            hists.append(np.asarray(chist, dtype=np.int))

        #print('Counter #%i [%i] (%i..%i/%i)' %(len(data),len(x),strt,i,len(content)))

        # Throw away first point; might have wrong counts
        if skip_first:
            x = x[1:]
            y = y[1:]
            y_err = y_err[1:]
            hists = hists[1:]
            if len(t)>0:
                t = t[1:]

        if sort:
            ind=np.argsort(x)

            x=x[ind]
            y=y[ind]
            y_err=y_err[ind]
            if len(t) == len(ind):
                t = t[ind]

            hists=[hists[ix] for ix in ind]

        counter = {'x':x, 'y':y, 'error':y_err, 'time':t, 'hists':hists}
        data.append(counter)

    return data, root

def read_xml(xml):
    #label = xml.attrib['label']
    sampling = xml.find('./sampling').text
    counter_node = xml.find('./counter')
    if counter_node is None:
        counter = ['0']
    else:
        counter = counter_node.text.split(',')


    timestamp = {}
    for it in xml.find('./timestamp'):
        for val in it:
            timestamp[it.tag,val.tag] = int(val.text)
    #print(timestamp)

    ionproperties = {}
    for it in xml.findall('./ionproperties/item'):
        ionproperties[it.attrib['name']] = float(it[0].text)
    #print(ionproperties)

    item = xml.find('./parameter/item')
    parameter = item.attrib
    for it in item:
        parameter[it.tag] = it.text
    #print(parameter)

    shims = {}
    for it in xml.findall('./shims/shim'):
        tmp = list()
        for val in it:
            tmp.append(float(val.text))
        shims[it.attrib['name']] = tmp
    #print(shims)

    profiles = {}
    for it in xml.findall('./profiles/profile'):
        tmp = {}
        #print('profile = ',it.attrib['name'])
        for val in it:
            #print('\tshim = ',val.attrib['name'])
            tmp[val.attrib['name']] = float(val.text)
        profiles[it.attrib['name']] = tmp
    #print(profiles)

    category = xml.find('./topname').text
    scriptname = xml.find('./scrname').text
    #print('%s/%s' % (category,scriptname))

    sourcecode = xml.find('./sourcecode').text
    if sourcecode is not None:
        if sourcecode.find('\\n'):
            sourcecode = sourcecode.replace('\\n','\n').replace('\\t','\t').replace('&lt;','<').replace('&gt;','>')
        else:
            sourcecode = sourcecode.replace(';',';\n').replace('{','{\n').replace('}','}\n')
    else:
        sourcecode = ''
    #print(sourcecode)
    return {'category':category, 'scriptname':scriptname, 'timestamp':timestamp,
            'counter':counter, 'sampling':sampling, 'parameter':parameter,
            'ionproperties':ionproperties, 'shims':shims, 'profiles':profiles, 'sourcecode':sourcecode}


path_data_local = '/home/qsim/Results/tiamo4.sync'
#path_data_local = 'smb://10.5.78.175/Paula_Data'
path_data_remote = '/mnt/qsim/tiamo4/messungen/EIOS'
# fname = eios_file.find_file(path_data_remote,'17_23_29',year='2018',month='05_Mai',day='08',cat='*',first=True)

def find_file(path,str_names,first=False,year='*',month='*',day='*',cat='*',scr='*',name='*.dat'):
    if type(str_names) is str:
        str_names = [str_names]
    #print(path+'/'+year+'/'+month+'/'+day+'/'+cat+'/'+scr+'/'+name)
    file_list = glob.glob(path+'/'+year+'/'+month+'/'+day+'/'+cat+'/'+scr+'/'+name, recursive=True)

    fname = list()
    for it_name in str_names:
        for it_file in file_list:
            if (it_file.find(it_name)>-1):
                fname.append(it_file)
                if first:
                    break
    return fname

def find_files(file_list, eios_data_path=path_data_local, year='*',month='*',day='*',cat='*',scr='*'):
    found = []
    months = {  '01':'Januar',    '02':'Februar',
                '03':'Maerz',     '04':'April',
                '05':'Mai',       '06':'Juni',
                '07':'Juli',      '08':'August',
                '09':'September', '10':'Oktober',
                '11':'November',  '12':'Dezember' }
    for fn in file_list:
        if len(fn)>18:
            year = fn[15:19]
            month = str(fn[12:14])+'_'+months[fn[12:14]]
            day = fn[9:11]
        fname = find_file(eios_data_path,fn,first=True,year=year,month=month,day=day,cat=cat,scr=scr,name='*.dat')
        found.extend(fname)
    return found

import json
import os
def save(filename,data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as open_file:
        json.dump(data, open_file, indent=4)

def load(filename):
    data = {}
    if os.path.exists(filename):
        with open(filename, "r") as open_file:
            data = json.load(open_file)
    return data

from time import localtime, strftime
def current_result_dir(local_dir=path_data_local, time_str=None):
    if time_str is None:
        time_str = strftime("%Y-%m-%d_%H-%M-%S", localtime())
    months = {  '01':'Januar',    '02':'Februar',
                '03':'Maerz',     '04':'April',
                '05':'Mai',       '06':'Juni',
                '07':'Juli',      '08':'August',
                '09':'September', '10':'Oktober',
                '11':'November',  '12':'Dezember' }
    date_list = time_str.split('_')[0].split('-')
    path = '%s/%s/%s_%s/%s'%(local_dir,date_list[0],date_list[1],months[date_list[1]],date_list[2])
    return path

#def load(filename,idx):
#    fn = './' + filename.split('/')[-1] + '_' + str(idx) + '.npz'
#    print(fn)
#    try:
#        npzfile = np.load(fn)
#    except: # ValueError or IOError
#        return False, [], [], []
#    return True, npzfile['x'], npzfile['y'], npzfile['y_err']

#def save(filename, idx, x, y, y_err):
#    fn = './' + filename.split('/')[-1] + '_' + str(idx) + '.npz'
#    np.savez(fn, x=x, y=y, y_err=y_err)
