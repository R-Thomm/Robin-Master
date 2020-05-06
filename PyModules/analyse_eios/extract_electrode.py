import eios_data as eios_file
import eios_analyse as eios_analyse

def quick_read(filename, idx = 0, sort=False, skip_first=True):
    success, x, y, y_err = eios_file.load(filename,idx)
    if not success:
        dat, xml = eios_file.read(filename, sort, skip_first)
        x = dat[idx]['x']
        #y = dat[idx]['y']
        #y_err = dat[idx]['error']
        hists = dat[idx]['hists']
        y, y_err = eios_analyse.fit_direct(hists, do_plot=True)
    eios_file.save(filename,idx,x,y,y_err)
    return x,y,y_err

def read_xml(fn):
    print('LOAD %s'%fn)
    _, xml = eios_file.read(fn, sort=False, skip_first=False)
    xml_dict = eios_file.read_xml(xml)
    return xml_dict

def find(path, key, day='*', month='*', year='*', cat='*', scr='*', first=False):
    file_list = eios_file.find_file(path, str_names=key, year=year, month=month, day=day, cat=cat, scr=scr, first=first)
    if first and len(file_list)>0:
        file_list = file_list[0]
    return file_list

from lxml import etree

def write_shim_xml(shims, fn_out='shim.xml'):
    root = etree.Element("shims")
    for i, (key, value) in enumerate(shims.items()):
        shim_entry = etree.SubElement(root, "shim", name=key, comment="")
        for j,value_shim in enumerate(value):
            shim_element = etree.SubElement(shim_entry, 'no%i'%j)
            shim_element.text = '%f'%value_shim

    et = etree.ElementTree(root)
    et.write(fn_out, encoding='UTF-8', pretty_print=True)
    print('WRITE %s'%fn_out)

    root_str = etree.tostring(root, pretty_print=True).decode("utf-8")
    return root_str

def write_profiles_xml(shims, fn_out='shim.xml'):
    root = etree.Element("profiles")
    for i, (key, value) in enumerate(profiles.items()):
        if i==0:
            profile = etree.SubElement(root, "profile", default="true", name=key)
        else:
            profile = etree.SubElement(root, "profile", default="false", name=key)
        for key_shim, value_shim in value.items():
            shim = etree.SubElement(profile, "shim", name=key_shim, value='%.5f'%value_shim)

    et = etree.ElementTree(root)
    et.write(fn_out, encoding='UTF-8', pretty_print=True)
    print('WRITE %s'%fn_out)

    root_str = etree.tostring(root, pretty_print=True).decode("utf-8")
    return root_str

def write_ionprop_xml(shims, fn_out='shim.xml'):
    root = etree.Element("ionproperties")
    for i, (key, value) in enumerate(ionprop.items()):
        shim = etree.SubElement(ionprop, "item", name=key, value='%.5f'%value)

    et = etree.ElementTree(root)
    et.write(fn_out, encoding='UTF-8', pretty_print=True)
    print('WRITE %s'%fn_out)

    root_str = etree.tostring(root, pretty_print=True).decode("utf-8")
    return root_str

if __name__ == "__main__":

    path = '/home/qsim/Results/tiamo4.sync'
    #path = '/home/bermuda/Results_H/tiamo3.sync'

    '''    
    scr = 'EURFMix_1P_Ex_Finer'
    keys = ['15_52_26_17_07_2018.dat',
            '20_35_35_16_10_2018.dat',
            '20_44_14_27_03_2019.dat']

    scr = '*'
    keys = ['17_02_39_08_10_2018.dat']
    '''
    scr = '*'
    #keys = ['17_04_47_09_05_2019.dat','17_17_01_09_05_2019.dat']
    keys = ['17_29_50_01_04_2020.dat']

    for key in keys:
        xml_dict = read_xml(find(path, key, scr=scr, first=True))
        
        name = key[:key.rfind('.')]
        shims = xml_dict['shims']
        shm_xml_str = write_shim_xml(shims, 'shim_%s.xml'%name)

        profiles = xml_dict['profiles']
        prf_xml_str = write_profiles_xml(profiles, 'shimprofiles_%s.xml'%name)

        #print(shm_xml_str,prf_xml_str)
        ionprop = xml_dict['ionproperties']
        prop_xml_str = write_ionprop_xml(ionprop,'ionprop_%s.xml'%name)
        

    '''
    for key, value in xml_dict.items():
        print (key, value)

    for key, value in profiles.items():
        print(key)

    for key, value in profiles['EU_squeeze'].items():
        print (key, value)

ionprop = xml_dict['ionproperties']
print(ionprop['EU_t_roc'])
'''

