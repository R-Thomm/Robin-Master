import torch

import numpy as np
import matplotlib.pyplot as plt

# dtype = torch.FloatTensor # Uncomment this to run on CPU
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

class CMyModel(torch.nn.Module):
    def __init__(self,limit, dim, NHIDDEN_CONV, CONV_KERNEL_SIZE):
        super(CMyModel, self).__init__()
        self.limit=limit
        self._CONV_KERNEL_SIZE = CONV_KERNEL_SIZE
        self.seq = torch.nn.Sequential(
                        #torch.nn.ReflectionPad1d(CONV_KERNEL_SIZE-1),
                        torch.nn.Conv1d(dim[0],NHIDDEN_CONV,CONV_KERNEL_SIZE,padding=0),
                        #torch.nn.Tanh(),
                        torch.nn.Conv1d(NHIDDEN_CONV,dim[1],CONV_KERNEL_SIZE,padding=0),
                        )

    def forward(self, x):
        #pad left (and right) with mean values of 5 first (last) values
        #this replaces "reflection pad"
        n_pad = self._CONV_KERNEL_SIZE-1
        pad_value_left = x[:,:,:5].mean()
        pad_value_right = x[:,:,-5:].mean()
        x_padded_left = torch.nn.functional.pad(x, (n_pad, 0), "constant", pad_value_left)
        x_padded = torch.nn.functional.pad(x_padded_left, (0, n_pad), "constant", pad_value_right)

        y_pred = self.seq(x_padded).clamp(-self.limit,self.limit)
        #y_pred = self.seq(x)
        return y_pred


class ConvWaveform(object):
    def __init__(self):
        self.best_x = None
        self.best_y = None
        self.best_d = np.inf

        self.init_model = False
        self.init_train = False

        self.time = []

    def initialize(self, limit=9.99, dim=[1,1], NHIDDEN_CONV=5, CONV_KERNEL_SIZE=41, time=[]):
        if self.init_model:
            raise Exception('Model already initalized!')
        self.limit = limit
        self.dim = dim
        self.NHIDDEN_CONV = NHIDDEN_CONV
        self.CONV_KERNEL_SIZE = CONV_KERNEL_SIZE

        self.time = time

        # instantiate ML model
        self.model = CMyModel(limit=self.limit, dim=self.dim, NHIDDEN_CONV=self.NHIDDEN_CONV, CONV_KERNEL_SIZE=self.CONV_KERNEL_SIZE)
        self.model.cuda()

        # select optimizer, loss function & target
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-3)
        self.loss_func = torch.nn.MSELoss()

        self.init_model = True

    def _get(self,x_numpy):
        #do experiment with this x and get real output
        x = self.FromShape(x_numpy)
        #print(x.shape, x_numpy.shape)
        y = self.ToShape(self.system(x),self.dim[0])
        dev = np.sqrt(np.mean((y-self.target)**2.,axis=-1))
        deviation = np.sum(dev,axis=-1)
        if deviation<self.best_d:
            self.best_x = np.copy(x_numpy)
            self.best_y = np.copy(y)
            self.best_d = deviation
        return y,deviation

    def _predict(self,y):
        y_tensor = self.ToTensor(y)
        x_tensor = self.model(y_tensor)
        x = self.ToNumpy(x_tensor)
        return x

    def learn(self, system, target, x_init = None, NITER = 100, N_TRAIN_ITER=15000, Nplot = 20, doplot=False):
        if not self.init_model:
            raise Exception('Model not initalized!')

        print('Model parameter number: %i ' % self.count_parameters())

        self.system = system
        self.target = self.ToShape(target,self.dim[0])
        self.target_tensor = self.ToTensor(target)
        
        if x_init is None:
            x_init = self._predict(self.target)
        else:
            x_init = self.ToShape(x_init,self.dim[1])

        # initialize accumulated training data: y, x, weights
        if not self.init_train:
            self.l_xtrain = x_init
            self.l_ytrain, self.l_wtrain = self._get(x_init)

        self.tloss = []
        self.tdev = []
        for i in range(NITER):
            Ntrains = int(N_TRAIN_ITER/(i+1))
            test_d, test_x, test_y, lloss = self._train(N_TRAIN_ITER = Ntrains)
            print("%i/%i: Current ChiSquared= %.2e Best= %.2e" % (i+1, NITER, test_d, self.best_d))

            self.tdev.append(test_d)            
            self.tloss.extend(lloss)
            if doplot and ((i==0) or (i%Nplot==0) or (i==NITER-1)):
                self._plot_funct(test_x, test_y, lloss, "current %.2e"%test_d)
        if doplot:
            self._plot_funct(self.best_x, self.best_y, self.tdev, "best %.2e"%self.best_d)

        return self.best_d, self.best_x, self.best_y

    def _train(self, N_TRAIN_ITER = 10000, BATCH_SIZE = 2,
                     N_TRAIN_MAX_BEST = 4, # max number of samples to train on (choosing best)
                     N_TRAIN_MAX_LAST = 0, # max number of samples to train on (choosing last)
                     ):
        if not self.init_model:
            raise Exception('Model not initalized!')

        self.model.train()
        
        #select data to train on, by selecting some of the last, and some of the best
        n_data = len(self.l_wtrain)
        i_train = np.arange(max(0,n_data-N_TRAIN_MAX_LAST), n_data)
        i_best = np.argsort(self.l_wtrain)
        for ithis in i_best:
            if len(i_train)>=N_TRAIN_MAX_BEST+N_TRAIN_MAX_LAST:
                break
            if ithis not in i_train:
                i_train=np.append(i_train, ithis)

        batch_n = min(len(i_train), BATCH_SIZE) #train also on few data
        N_batches = len(i_train)//batch_n
        lloss=[]
        if batch_n>0:
            train_y = self.ToTensor(self.l_ytrain[i_train])
            train_x = self.ToTensor(self.l_xtrain[i_train])
            for k in range(N_TRAIN_ITER):
                #shuffle data
                shuffle_i = np.random.choice(len(i_train),len(i_train),False)

                sloss=0.
                for j in range(N_batches):
                    #first: train the model on collected data:
                    self.model.zero_grad()

                    select = shuffle_i[j*batch_n:(j+1)*batch_n]
                    #print(select)
                    shuffled_batch_y = train_y[select]
                    shuffled_batch_x = train_x[select]

                    predict_x = self.model(shuffled_batch_y)
                    loss = self.loss_func(predict_x,shuffled_batch_x,)
                    loss.backward()
                    self.optimizer.step()
                    sloss+=loss.item()
                lloss.append(sloss/N_batches)
                    
        self.model.eval()

        #predict new data
        x = self._predict(self.target)
        y,d = self._get(x)
        

        #add to training data
        '''
        y_tensor = self.ToTensor(y)
        self.l_ytrain.append(y_tensor)
        self.l_xtrain.append(x_tensor)
        self.l_wtrain.append(d)
        '''
        self.l_ytrain = np.append(self.l_ytrain, y, axis=0)
        self.l_xtrain = np.append(self.l_xtrain, x, axis=0)
        self.l_wtrain = np.append(self.l_wtrain, d, axis=0)

        return d, x, y, lloss
    
    def predict(self,y):
        if not self.init_model:
            raise Exception('Model not initalized!')
        y_shape = self.ToShape(y,self.dim[0])
        data = self.FromShape(self._predict(y_shape))
        return data

    def save(self,path='./model.torch'):
        if not self.init_model:
            raise Exception('Model not initalized!')

        #print("Model's state_dict:")
        #for param_tensor in self.model.state_dict():
        #    print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        #print("Optimizer's state_dict:")
        #for var_name in self.optimizer.state_dict():
        #    print(var_name, "\t", self.optimizer.state_dict()[var_name])

        state_dict = dict()

        state_dict['limit'] = self.limit
        state_dict['dim'] = self.dim
        state_dict['NHIDDEN_CONV'] = self.NHIDDEN_CONV
        state_dict['CONV_KERNEL_SIZE'] = self.CONV_KERNEL_SIZE

        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()

        state_dict['ytrain'] = self.l_ytrain
        state_dict['xtrain'] = self.l_xtrain
        state_dict['wtrain'] = self.l_wtrain

        state_dict['best_dev'] = self.best_d
        state_dict['best_x'] = self.best_x
        state_dict['best_y'] = self.best_y

        state_dict['time'] = self.time

        torch.save(state_dict, path)

    def load(self,path='./model.torch'):
        state_dict = torch.load(path)
        
        limit = state_dict['limit']
        dim = state_dict['dim']
        NHIDDEN_CONV = state_dict['NHIDDEN_CONV']
        CONV_KERNEL_SIZE = state_dict['CONV_KERNEL_SIZE']

        self.initialize(limit,dim,NHIDDEN_CONV,CONV_KERNEL_SIZE)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

        self.l_ytrain = state_dict['ytrain']
        self.l_xtrain = state_dict['xtrain']
        self.l_wtrain = state_dict['wtrain']

        self.best_d = state_dict['best_dev']
        self.best_x = state_dict['best_x']
        self.best_y = state_dict['best_y']

        self.time = state_dict['time']

        self.model.eval()

        self.init_train = True

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def ToShape(self,numpy_data,dim):
        return numpy_data.reshape(1,dim,-1)

    def FromShape(self,numpy_data):
        #return np.squeeze(numpy_data)
        return numpy_data[0,:]

    def ToTensor(self,numpy_data):
        return torch.from_numpy(numpy_data).type(dtype)

    def ToNumpy(self,tensor_data):
        #return tensor_data.detach().numpy()
        return tensor_data.cpu().detach().numpy()

    def _plot_funct(self,x,y,loss,title):
        x = self.FromShape(x)
        y = self.FromShape(y)
        y0 = self.FromShape(self.target)
        dy = y-y0
        
        print('loss %.2e'% (loss[-1]))
        #print(x.shape,y.shape,y0.shape,dy.shape)
        
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,8), sharex=False)
        f.suptitle(title)
        #print(x.shape, y.shape)
        for i in range(x.shape[0]):
            ax3.plot(x[i,:],'x-', label='input %i'%i)
        for i in range(y.shape[0]):
            ax1.plot(y[i,:],'x-', label='system output %i'%i)
            ax1.plot(y0[i,:],'x-', label='target output %i'%i)
            ax2.plot(dy[i,:],'x', label='deviation %i'%i)
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax3.set_xlabel('Time in us')

        ax4.plot(loss)
        ax4.set_yscale('log')
        ax4.set_ylabel('loss')
        ax4.set_xlabel('train epoch')
        plt.show()
        
def model_evaluator(name_model):
    cmodel = ConvWaveform()
    cmodel.load(name_model)

    time = np.array(cmodel.time)
    X = np.array(cmodel.best_x)
    predictor = lambda x: np.array(cmodel.predict(x))

    return time, X, predictor

