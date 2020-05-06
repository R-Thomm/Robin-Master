import numpy as np
import matplotlib.pyplot as plt
import json
import os.path

import sys
sys.path.insert(0, './../../')

from scipy.interpolate import interp1d
from scipy.interpolate import Rbf

from PyModules.machine_learning.interface import Interface
from PyModules.machine_learning.model_torch import ConvWaveform
from PyModules.pdq.waveform import smooth_step_eu, b_ex_1, b_ex_2, d_b_ex_2, linear_ramp
from PyModules.pdq.waveform import resample, erfc_step, fit_curve

def write_waveform(func, name_model, name_wvf):
    from PyModules.pdq.pdq_waveform import save_wvf
    from PyModules.machine_learning.model_torch import model_evaluator
    time, X, predictor = model_evaluator(name_model)
    
    T = (time[-1]-time[0])
    t0 = time[0]+T/2.
    y = func(time, t0)
    x = predictor(y)
    t = time*1e-6

    #data_wf = X
    data_wf = x.reshape((x.shape[0],1,x.shape[1]))
    data_wf = np.insert(data_wf, data_wf.shape[0], x[0,0,], axis=0)
    #print(t.shape, x.shape, y.shape, data_wf.shape)

    # write waveform to pdq file
    save_wvf(t, data_wf=data_wf, data_file=name_wvf)

    return time, x, y

class System(object):
    def __init__(self,N,T):
        # waveform sampling & timings
        self.N = N
        self.T = T
        self.t = np.linspace(0.,self.T,num=self.N,endpoint=False)
        # PDQ & osci channel
        self.ch_osci = [1,2] #[1,2] [3,2]
        self.index_control = 1
        self.index_signal = 0
        # PDQ parameter
        self.pdq_address = ['DREIECK2']
        self.pdq_ch_per_stck = 9
        self.multiplier = False
        self.interp_order = 0
        self.ch_pdq = np.array([ 0,0,0,  0,0,0,  0,1,1 ])
        # Init class values
        self.t_shift = [0]*len(self.ch_osci)
        # store all measurements
        self.storage = { 'N':self.N, 'T':self.T, 
                         't_pdq':self.t.tolist(), 'data_pdq':[], 
                         't_osci':[], 'data_osci':[] } 
        # Init interface EIOS, PDQ & Osci
        self.exp = Interface(serial_numbers=self.pdq_address, pdq_ch_per_stck=self.pdq_ch_per_stck, multiplier=self.multiplier)

    def __del__(self):
        del self.exp

    def get(self,data_in,keep=True):
        # add control channel
        #print(data_in.shape)

        data_shp = data_in.shape
        data_pdq = data_in.reshape((data_shp[0],1,data_shp[1]))

        #print(data_pdq.shape)
        data_pdq = np.insert(data_pdq, data_pdq.shape[0], data_in[0,], axis=0)
        #print(data_pdq.shape)
        
        # ... and initial voltages
        data_init = np.zeros(self.ch_pdq.shape)
        data_init[self.ch_pdq>0] = data_pdq[:,0,0]

        # time vector in pdq units
        times = self.t*1e-6

        # measure
        while True:
            t_osci,data_osci = self.exp.measure(times,data_pdq,data_init,self.ch_pdq,self.ch_osci,interp_order=self.interp_order)
            if (len(data_osci.shape)>1) and (data_osci.shape[1]>1):
                break
            print('Repeat: osci trigger fail (data empty): ',data_osci.shape,len(data_osci.shape))
            
        # save input & output data
        if keep:
            self.storage['data_pdq'].append(data_in.tolist())
            self.storage['t_osci'].append(t_osci.tolist())
            self.storage['data_osci'].append(data_osci.tolist())

        # prepare data: time in us
        t_o = t_osci*1e6

        return t_o, data_osci


    def measure(self,data_in):
        t_o, data_osci = self.get(data_in)

        # select PD channel
        data_o = np.array([data_osci[self.index_signal,:]])
        #data_o = np.array(data_osci[:2,:])
        #data_o[1,:] = data_o[1,:]/32.

        # resample osci data to waveform timings
        f = interp1d(t_o, data_o)
        #f = Rbf(t_o, dat, smooth=5)
        data_out = f(self.t)


        #f = interp1d(t_o, data_o, axis=1)
        #data_out = f(self.t-t_delay)

        return data_out

    def save(self,filename):
        with open(filename, "w") as write_file:
            json.dump(self.storage, write_file)
            #json.dump(self.storage, write_file, indent=4)

if __name__ == "__main__":
    # waveform timings
    T = 2*5.28;
    N = 66*int(T/5.28) # 66 125 250
    # ML parameter
    NITER = 2 # 400
    N_TRAIN_ITER = 800
    NHIDDEN_CONV=17
    CONV_KERNEL_SIZE=42
    Nplot = 15
    save_data = False
    load_data = False
    use_cali = False
    file_model = '../UserData/Calibration/model_ml.json'

    if len(sys.argv)>1:
        N = int(sys.argv[1])
    if len(sys.argv)>2:
        T = float(sys.argv[2])
    if len(sys.argv)>3:
        NITER = int(sys.argv[3])
    if len(sys.argv)>4:
        N_TRAIN_ITER = int(sys.argv[4])
    if len(sys.argv)>5:
        NHIDDEN_CONV = int(sys.argv[5])
    if len(sys.argv)>6:
        CONV_KERNEL_SIZE = int(sys.argv[6])
    if len(sys.argv)>7:
        Nplot = bool(sys.argv[7])
    if len(sys.argv)>8:
        save_data = bool(sys.argv[8])
    if len(sys.argv)>9:
        load_data = bool(sys.argv[9])
    if len(sys.argv)>10:
        use_cali = bool(sys.argv[10])

    print('N=%i, T=%f'%(N,T))
    print('NITER=%i, N_TRAIN_ITER=%i, N_hidden=%i, N_kernel=%i, Nplot=%i'%(NITER,N_TRAIN_ITER,NHIDDEN_CONV,CONV_KERNEL_SIZE,Nplot))
    print('save %i, load %i, use calibration %i'%(save_data,load_data,use_cali))
    #sys.exit(0)

    #System
    SYS = System(N,T)
    t = SYS.t

    # waveform (target) data
    if use_cali:
        A = 2.87; dip_depth = 1.; t_width = 0.25
        y_w = b_ex_2(t,A,dip_depth,t_width,T/2.)
        '''
        # Atten. control
        A_low = .5; A_high = 2.81; t_width = .2; 
        t_hold = 1.; t_center = T/2; t_l = t_center-t_hold/2.; t_r = t_center+t_hold/2.
        y_w = (smooth_step_eu(-t, -t_l, t_width) + smooth_step_eu(t, t_r, t_width))*(A_high-A_low) + A_low
        '''
        # Calculate target from calibration
        from cloudpickle import dump, load
        with open('../UserData/Calibration/cali_func.pkl', 'rb') as f: 
            cali_data = load(f)
        # calibration functions
        func_mf = cali_data['f_mf']
        func_PD = cali_data['f_mf2pd']
        func_i_mf = cali_data['f_i_mf']
        func_i_hf = cali_data['f_i_hf']
        # initial guess
        x_wf = func_i_mf(y_w)
        #A_high = func_i_mf(y_w[0])
        #A_low = func_i_mf(A-dip_depth)
        #x_wf = np.array([A_high]*N)
        #x_wf[int(N/2-dx):int(N/2+dx)] = A_low
    else:
        # WF
        t_width = .5
        
        # PD pickup signal
        A = 656.2e-3; dip_depth = 656.2e-3-710.0e-3
        y_w_1 = b_ex_2(t,A,dip_depth,t_width,T/2.)

        # PD reflection signal
        A_r = 1.616; dip_depth_r = A_r-1.22;
        #y_w_2 = d_b_ex_2(t+.1,A_r,dip_depth_r,t_width,T/2.)/32.
        #y_w = np.array([y_w_1, y_w_2])
        y_w = np.array([y_w_1])

        # dummy calibration functions
        func_mf = lambda x: x
        func_PD = lambda x: x
        # initial guess
        # Atten. control
        A_high = 3.141; A_low = 2
        dx = N*t_width/T
        x_wf = np.array([A_high]*N)
        x_wf[int(N/2-dx):int(N/2+dx)] = A_low

    # target waveform
    y0 = func_PD(y_w)
    dim = [1,1]
    target_out = y0.reshape((1,dim[0],-1))
    
    print('target shape',target_out.shape,'dim',dim)
    # initial guess waveform
    x_init = np.array([x_wf])
    #x_init = np.array([x_wf,[0.]*len(x_wf)])
    
    '''
    # plot target and initial guess
    plt.plot(t,target_out[0,:])
    plt.plot(t,x_init[0,:])
    plt.plot(t,x_wf)
    plt.show()
    '''
    '''
    # correct osci trigger delays 
    offset = np.max(x_wf)
    control_amp = np.std(x_wf)/2.
    width = 4.*t_width
    SYS.find_delay(control_amp = control_amp, width = width, offset = offset)
    '''
    
    # train & predict & save model
    cmodel = ConvWaveform()
    
    
    if os.path.isfile(file_model) and load_data:
        cmodel.load(file_model)
    else:
        cmodel.initialize(dim=dim, NHIDDEN_CONV=NHIDDEN_CONV, CONV_KERNEL_SIZE=CONV_KERNEL_SIZE, time=t)

    data_dev,data_x,data_y = cmodel.learn(SYS.measure,
                                target_out, x_init=x_init, 
                                NITER=NITER, N_TRAIN_ITER=N_TRAIN_ITER, 
                                Nplot=Nplot, doplot=False)
    #cmodel.tloss
    #cmodel.tdev
    print('best chi %.2e'%cmodel.best_d)
    devis = []
    data_col = []
    target_in = cmodel.predict(target_out)
    N_avg = 10
    for i in range(N_avg):
        data_out = SYS.measure(target_in)
        #print(data_out.shape)
        #devi = np.sqrt(np.mean((data_out-target_out[0,])**2.),axis=2)
        devi = np.sqrt(np.mean((data_out-target_out[0,])**2.))
        devis.append(devi)
        data_col.append(data_out)
        print('%.2e'%devi)

    devis = np.array(devis)
    data_col = np.array(data_col)
    data_avg = np.mean(data_col,axis=0)
    data_std = np.std(data_col,axis=0)/np.sqrt(N_avg)
    print('mean %.2e, std %.2e'%(np.mean(devis), np.std(devis)))
    plt.errorbar(t,data_avg[0,:],yerr=data_std[0,:])
    plt.show()

    # save this run to file
    if save_data:
        # model
        cmodel.save(file_model)
        # osci & pdq data
        from time import localtime, strftime
        time_str = strftime("%Y-%m-%d_%H-%M-%S", localtime())
        file_data = '../UserData/Calibration/data_ml_%s.json' % time_str
        SYS.save(file_data)

    # Plot best run
    #f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    #for i in range(len(data_x)):
    #    ax1.plot(t,target_out[i,:],'--',label='target %i'%i)
    #    ax1.plot(t,data_y[i,:],'.-',label='output %i'%i)
    #    ax2.plot(t,data_x[i,:],'.-',label='input %i'%i)
    #    ax3.plot(t,y_w,'.-',label='soll %i'%i)
    #    ax3.plot(t,func_mf(data_y[i,:]),'.-',label='haben %i'%i)

    #A = np.max(y0)
    #B = np.min(y0)
    #ax1.axhline(y=B, color='c', linestyle='--', label='base %.2f'%B)
    #ax1.axhline(y=A, color='r', linestyle='--', label='atten %.2f'%A)
    #ax1.legend()
    #ax2.legend()
    #ax3.legend()
    #plt.show()

