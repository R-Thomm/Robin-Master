import numpy as np
import matplotlib.pyplot as plt

from osci import Oscilloscope
#osci = Oscilloscope()
osci = Oscilloscope(fast=True)
usb = osci.list_device()
adr = usb[-1]
for name in usb:
    if (name.find('DS1Z')>-1):
        adr = name
        break
print('Oscilloscope Address\n\t%s'%adr)
osci.open(adr)

channel=[1,2,3,4]
N = 10

n = 0
data_cum = []
data_sum = np.zeros((4,1200))
thres = .25 #.0007 #.09 #.006 #.003 #.25
ch_idx = 1
i = 0
while n<N:
    t, data, dt = osci.read(channel)
    
    #data_sum += data
    #n +=1

    dev = np.std(data[ch_idx])
    print('#%i/%i'%(n,N),dev,(dev>thres))
    if dev>thres:
        data_sum += data
        data_cum.append(data.tolist())
        n += 1
    else:
        i += 1

    if n<1 and i>N:
        print('Timeout: %i success / %i attempts'%(n,i))
        break

if n>0:
    data_avg = data_sum/n

    data_json = dict()
    data_json['t'] = t.tolist()
    data_json['data'] =  data_cum
    data_json['data_avg'] =  data_avg.tolist()

    x = t*1e6
    plt.plot(x,data_avg[0],label='PD AO')
    plt.plot(x,data_avg[1],label='control')
    plt.plot(x,data_avg[2],label='PD refl.')
    plt.plot(x,data_avg[3],label='TTL')
    plt.legend()
    plt.xlabel('Time in us')
    plt.ylabel('Voltage in V')
    plt.show()

    from time import localtime, strftime
    filename = './data/trace_%s.json' % (strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    print('Save %s'%filename)

    import json
    with open(filename, "w") as write_file:
        json.dump(data_json, write_file, indent=4)

