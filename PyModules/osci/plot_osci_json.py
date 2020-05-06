import json
import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":

    #fn0 = '../UserData/Calibration/data_ml_2019-03-05_11-14-32.json'
    #fn0 = '../UserData/Calibration/data_ml_2019-03-05_11-41-03.json'
    fn0 = '../UserData/Calibration/data_ml_2019-03-06_09-20-32.json'
    #fn0 = '../UserData/Calibration/data_ml_2019-03-07_11-11-59.json'

    if len(sys.argv)>1:
        fn0 = sys.argv[1]

    with open(fn0, "r") as read_file:
        data0 = json.load(read_file)

    t_pdq = np.array(data0['t_pdq'])
    t_osci = np.array(data0['t_osci'][0])*1e6

    data_pdq = np.array(data0['data_pdq'])
    data_osci = np.array(data0['data_osci'])

    '''
    for d_p,d_o in zip(data_pdq,data_osci):
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ax1.plot(t_pdq,d_p[0,:],label='PDQ input')
        ax1.legend()

        ax2.plot(t_osci,d_o[0,:],label='Osci CH1 - PD')
        ax2.plot(t_osci,d_o[1,:],label='Osci CH2 - control')
        ax2.legend()

        plt.show()
    '''

    from matplotlib.animation import FuncAnimation

    N = len(data_pdq)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ln0, = ax1.plot([],[],label='PDQ input')
    ln1, = ax2.plot([],[],label='Osci CH1 - PD')
    ln2, = ax1.plot([],[],label='Osci CH2 - control')

    def init():
        ax1.legend()
        ax2.legend()

        ax1.set_xlim(0, np.max(t_pdq))
        ax2.set_xlim(0, np.max(t_osci))

        y_min_10 = np.min(data_pdq)
        y_max_10 = np.max(data_pdq)

        y_min_11 = np.min(data_osci[:,1,:])
        y_max_11 = np.max(data_osci[:,1,:])

        y_min_1 = np.min([y_min_10,y_min_11])-.5
        y_max_1 = np.max([y_max_10,y_max_11])+.5

        y_min_2 = np.min(data_osci[:,0,:])-.05
        y_max_2 = np.max(data_osci[:,0,:])+.05

        ax1.set_ylim(y_min_1, y_max_1)
        ax2.set_ylim(y_min_2, y_max_2)

        return ln0,ln1,ln2,

    def update(idx):
        fig.suptitle('Set %i/%i'%(idx,N-1),fontsize=14)

        data0 = data_pdq[idx,:][0,:]
        data1 = data_osci[idx,:][0,:]
        data2 = data_osci[idx,:][1,:]

        #print(data0.shape, data1.shape)

        ln0.set_data(t_pdq,data0)
        ln1.set_data(t_osci,data1)
        ln2.set_data(t_osci,data2)

        #plt.draw()

        return ln0,ln1,ln2,

    ani = FuncAnimation(fig, update, frames=range(1,N),
                        init_func=init, blit=False)
    plt.show()
