import numpy as np
import matplotlib.pyplot as plt
from model_torch import ConvWaveform

# load model and predict waveform
cmodel = ConvWaveform()
fname_model = '../UserData/Calibration/model_ml.json'
cmodel.load(fname_model)
x = cmodel.best_x
times = cmodel.time
#target_in = cmodel.predict(target_out)

data_wf = x.reshape((x.shape[0],1,x.shape[1]))
data_wf = np.insert(data_wf, data_wf.shape[0], x[0,], axis=0)

print(data_wf.shape)

plt.plot(x[0,:],'x--')
plt.show()

shim = np.array([ 0.,0.,0.,  0.,0.,0.,  0.,1.,1. ])
voltage_init = np.zeros_like(shim)

# write waveform to pdq file
from pdq.pdq_waveform import PDQ_Waveform
data_file='../UserData/waveform_13.json'

pdq_wave = PDQ_Waveform(serial_numbers=['DREIECK2'], pdq_ch_per_stck = 9, multiplier = True)
pdq_wave.save(times, data_wf=data_wf, data_file=data_file)
#pdq_wave.send_file(shim,voltage_init,data_file)

