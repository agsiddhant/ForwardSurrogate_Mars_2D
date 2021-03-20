import numpy as np
import keras
from joblib import load
import glob, os
import warnings
from time import sleep
from keras.callbacks import Callback
from joblib import load, dump, Parallel, delayed
import time
from keras.layers import Layer
from keras import backend as K
import random

import tensorflow as tf

from keras.layers import InputLayer, Reshape, Dense, Dropout, LSTM, ConvLSTM2D, Conv3D, Activation
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.utils import multi_gpu_model 
from keras import regularizers
from keras.callbacks import History 
from keras.constraints import maxnorm
from keras.layers.normalization import BatchNormalization

from scipy.fft import dct, idct
from tensorflow.python.keras.utils.data_utils import Sequence

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, pathSave, sVar, trainOrVal='train', batch_size=256, xdim=(302,394), ydim=(302,394), n_channels=1,
                 shuffle=True, isAutoencoder=False):
        'Initialization'
        self.xdim = xdim
        self.ydim = ydim
        self.batch_size = batch_size
        self.trainOrVal = trainOrVal
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.pathSave = pathSave
        self.sVar = sVar
        self.numSims = [12000]
        self.isAutoencoder = isAutoencoder
        dictToOpen = pathSave + "processedDicts/Mars_new/" + sVar + "/Dict_NewMars_preProcessed_" + str(self.numSims) + "sims_0.9trPercent.txt"
        with open(dictToOpen, "rb") as myFile:
            indexer = load(myFile)
        self.indices = indexer["Indexer_" + trainOrVal]
        self.pMax = indexer["pMax"]
        self.pMin = indexer["pMin"]
        print(self.pMin,self.pMax)
        dictToOpen = pathSave + "processedDicts/Mars_new/T/Dict_NewMars_preProcessed_" + str(self.numSims) + "sims_0.98trPercent.txt"
        with open(dictToOpen, "rb") as myFile:
            indexer = load(myFile)
        self.numBatches = int(len(self.indices)/self.batch_size)
        self.on_epoch_end()
                                  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numBatches
                                  
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = self.indexes[index]
        
        # Generate data
        #t0 = time.time()
        X = self.__data_generation(index)
        #t1 = time.time()
        #print(t1-t0)
        #X = dct(X, type=2, norm='ortho')
        X = (X-self.pMin)/(self.pMax-self.pMin)
        return X, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numBatches)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batchN):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,self.xdim[0],self.xdim[1],self.n_channels))      
        
        def makeMatrix(ind,i):  
            try:
                with open(self.pathSave + "processedDicts/Mars_new/ConvAE/" + self.sVar + "/train/" + str(self.indices[i][0]) + "/Dict_processed_data_MarsNew_2D_ConvAE_" + str(self.indices[i][2]) + ".txt", "rb") as myFile:
                        X[ind,:,:,:] = load(myFile)
            except Exception as e:
                print(e)
                    
        Parallel(n_jobs=4,prefer='threads')(delayed(makeMatrix)(i_,v_) for i_,v_ in enumerate(range(self.batch_size*batchN,self.batch_size*(batchN+1))))
        return X
    
class DataGenerator_uv(Sequence):
    'Generates data for Keras'
    def __init__(self, pathSave, sVar, trainOrVal='train', batch_size=256, xdim=(302,394), ydim=(302,394), n_channels=1,
                 shuffle=True, isAutoencoder=False):
        'Initialization'
        self.xdim = xdim
        self.ydim = ydim
        self.batch_size = batch_size
        self.trainOrVal = trainOrVal
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.pathSave = pathSave
        self.sVar = sVar
        self.numSims = [12000]
        self.isAutoencoder = isAutoencoder
        dictToOpen = pathSave + "processedDicts/Mars_new/T/Dict_NewMars_preProcessed_" + str(self.numSims) + "sims_0.98trPercent.txt"
        with open(dictToOpen, "rb") as myFile:
            indexer = load(myFile)
        self.indices = indexer["Indexer_" + trainOrVal]
        
        dictToOpen = pathSave + "processedDicts/Mars_new/" + self.sVar + "/Dict_NewMars_preProcessed_" + str(self.numSims) + "sims_0.9trPercent.txt"
        with open(dictToOpen, "rb") as myFile:
            indexer = load(myFile)
        self.oMax = indexer["pMax"]
        self.oMin = indexer["pMin"]
        
        
        dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_[12000]sims_1channel_ae840_0p98tr_wt.txt" 
        with open(pathSave + dictToOpen, "rb") as myFile:
            Dict_processed_data = load(myFile)

        self.pMin = Dict_processed_data['paraMin'] 
        self.pMax = Dict_processed_data['paraMax']
        print(self.oMin,self.oMax)
        self.numBatches = int(len(self.indices)/self.batch_size)
        self.on_epoch_end()
                                  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numBatches
                                  
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = self.indexes[index]
        
        # Generate data
        #t0 = time.time()
        X, Y = self.__data_generation(index)
        #t1 = time.time()
        #print(t1-t0)
        Y = (Y-self.oMin)/(self.oMax-self.oMin)
        for i in range(6):
            X[:,i] = (X[:,i]-self.pMin[i])/(self.pMax[i]-self.pMin[i])
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numBatches)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batchN):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        Y = np.zeros((self.batch_size,self.xdim[0],self.xdim[1],self.n_channels))
        X = np.zeros((self.batch_size,6))
        
        for ind,i in enumerate(range(self.batch_size*batchN,self.batch_size*(batchN+1))): 
            try:
                with open(self.pathSave + "processedDicts/Mars_new/ConvAE/" + self.sVar + "/train/" + str(self.indices[i][0]) + "/Dict_processed_data_MarsNew_2D_ConvAE_" + str(self.indices[i][2]) + ".txt", "rb") as myFile:
                    Y[ind,:,:,0:1] = load(myFile)
                with open(self.pathSave+"processedDicts/Mars_new/ConvAE/" + self.sVar + "/train/" + str(self.indices[i][0]) + "/paras.txt", "rb") as myFile:
                    X[ind,1:] = load(myFile)  
                X[ind,0] = self.indices[i][2]
            except Exception as e: 
                print(e)
        return X, Y
    
class DataGenerator_lstm(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pathSave, numSims,  trainOrVal='train', batch_size=256, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.trainOrVal = trainOrVal
        self.shuffle = shuffle
        self.pathSave = pathSave
            
        if trainOrVal=='train':
            dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_9877sims_1channel_ae840_x_data_0p98tr_wt.txt" 
        else:
            dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_107sims_1channel_ae840_x_cv_0p98tr_wt.txt" 
        with open(pathSave + dictToOpen, "rb") as myFile:
            self.x_data = load(myFile)
        
        self.encoding_size = self.x_data.shape[2]*self.x_data.shape[3]*(self.x_data.shape[4]-1)
        self.input_size = 8
        
        print(trainOrVal + ' data: ' + str(self.x_data.shape))
        self.timeSteps = 20
        self.indices = []
        for s in range(numSims):
            for t in range(1,467+1):
                seq = np.arange(t-self.timeSteps,t)
                for sind,sval in enumerate(seq):
                    if sval<0:
                        seq[sind] = 0
                if self.x_data[s,seq[-1],0,0,0] > 0:
                    self.indices.append([s, seq])
        self.numBatches = int(len(self.indices)/self.batch_size)
        self.on_epoch_end()
                                  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numBatches
                                  
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_ = self.indexes[index]
        
        # Generate data
        #t0 = time.time()
        X, Y = self.__data_generation(index_)
        #t1 = time.time()
        #print(t1-t0)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numBatches)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batchN):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,self.timeSteps,self.encoding_size+self.input_size))    
        Y = np.zeros((self.batch_size,self.encoding_size))
        for i,v in enumerate(range(self.batch_size*batchN,self.batch_size*(batchN+1))):
            X[i,:,self.input_size:] = self.x_data[self.indices[v][0],self.indices[v][1],:,:,1:].reshape(1,self.timeSteps,self.encoding_size)
            X[i,:,0:7] = self.x_data[self.indices[v][0],self.indices[v][1],0,0:7,0]
            X[i,:,7] = self.x_data[self.indices[v][0],self.indices[v][1],1,0,0]
            Y[i,:] = self.x_data[self.indices[v][0],self.indices[v][1][-1]+1,:,:,1:].reshape(1,self.encoding_size)
        return X, Y
    
class DataGenerator_lstm_full(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pathSave, numSims,  trainOrVal='train', batch_size=256, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.trainOrVal = trainOrVal
        self.shuffle = shuffle
        self.pathSave = pathSave
        dictToOpen = pathSave + "processedDicts/Mars_new/T/Dict_NewMars_preProcessed_[12000]sims_0.9trPercent.txt"
        with open(dictToOpen, "rb") as myFile:
            pmaxmin = load(myFile)
        self.oMax = pmaxmin["pMax"]
        self.oMin = pmaxmin["pMin"]
            
        if trainOrVal=='train':
            dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_9877sims_1channel_ae840_x_data_0p98tr_wt.txt" 
        else:
            dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_107sims_1channel_ae840_x_cv_0p98tr_wt.txt" 
        with open(pathSave + dictToOpen, "rb") as myFile:
            self.x_data = load(myFile)
        
        self.encoding_size = self.x_data.shape[2]*self.x_data.shape[3]*(self.x_data.shape[4]-1)
        self.input_size = 8
        
        print(trainOrVal + ' data: ' + str(self.x_data.shape))
        self.timeSteps = 20
        self.indices = []
        for s in range(numSims):
            for t in range(1,467+1):
                seq = np.arange(t-self.timeSteps,t)
                for sind,sval in enumerate(seq):
                    if sval<0:
                        seq[sind] = 0
                if self.x_data[s,seq[-1],0,0,0] > 0:
                    self.indices.append([s, seq])
        self.numBatches = int(len(self.indices)/self.batch_size)
        self.on_epoch_end()
        
        dictToOpen = pathSave + "processedDicts/Mars_new/T/Dict_NewMars_preProcessed_[12000]sims_0.98trPercent.txt"
        with open(dictToOpen, "rb") as myFile:
            indexMapper = load(myFile)
        indexMap = indexMapper["Indexer_" + trainOrVal]
        indexMap = [tuple(s) for s in indexMap]
        dtype = [('dict', int), ('useless', int), ('time', float)]
        indexMap = np.array(indexMap, dtype=dtype)
        indexMap = np.sort(indexMap, order=['dict', 'time']) 
        indexMap = indexMap.tolist()
        self.simList = []
        self.timeList = []
        dictInd = -1
        for ind,val in enumerate(indexMap):
            if val[0] != dictInd:
                if dictInd != -1:
                    self.timeList.append(innertimeList)
                    self.simList.append(dictInd)
                dictInd = val[0]
                innertimeList = [val[2]]
            else:
                innertimeList.append(val[2])    

        self.simList.append(dictInd)
        self.timeList.append(innertimeList)
            
        sindices = np.arange(self.x_data.shape[0])       
        random.seed(6)
        random.shuffle(sindices)
        self.simList = [self.simList[s] for s in sindices]
        self.timeList = [self.timeList[s] for s in sindices]
                                  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numBatches
                                  
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_ = self.indexes[index]
        
        # Generate data
        #t0 = time.time()
        X, Y = self.__data_generation(index_)
        #t1 = time.time()
        #print(t1-t0)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numBatches)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batchN):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,self.timeSteps,self.encoding_size+self.input_size))    
        Y = np.zeros((self.batch_size,302,394,1))
        
        def makeMatrix(i,v):  
            try:
                X[i,:,self.input_size:] = self.x_data[self.indices[v][0],self.indices[v][1],:,:,1:].reshape(1,self.timeSteps,self.encoding_size)
                X[i,:,0:7] = self.x_data[self.indices[v][0],self.indices[v][1],0,0:7,0]
                X[i,:,7] = self.x_data[self.indices[v][0],self.indices[v][1],1,0,0]
                simind = self.simList[self.indices[v][0]]
                timind = self.timeList[self.indices[v][0]][self.indices[v][1][-1]+1]
                with open(self.pathSave + "processedDicts/Mars_new/ConvAE/T/train/" + str(simind) + "/Dict_processed_data_MarsNew_2D_ConvAE_" + str(timind) + ".txt", "rb") as myFile:
                    field = load(myFile)
                field = (field-self.oMin)/(self.oMax-self.oMin)             
                Y[i,:,:,:] = field
            except Exception:
                pass
            
                    
        Parallel(n_jobs=4,prefer='threads')(delayed(makeMatrix)(i_,v_) for i_,v_ in enumerate(range(self.batch_size*batchN,self.batch_size*(batchN+1))))
        
        return X, Y
    
class DataGenerator_conv_lstm(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, pathSave, numSims,  trainOrVal='train', batch_size=256, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.trainOrVal = trainOrVal
        self.shuffle = shuffle
        self.pathSave = pathSave
            
        if trainOrVal=='train':
            dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_9877sims_1channel_ae840_x_data_0p98tr_wt.txt" 
        else:
            dictToOpen = "processedDicts/Mars_new/ConvLSTM/T/Dict_processed_data_MarsNew_2D_ConvLSTM_T_107sims_1channel_ae840_x_cv_0p98tr_wt.txt" 
        with open(pathSave + dictToOpen, "rb") as myFile:
            self.x_data = load(myFile)
        
        print(trainOrVal + ' data: ' + str(self.x_data.shape))
        self.timeSteps = 20
        self.indices = []
        for s in range(numSims):
            for t in range(1,467+1):
                seq = np.arange(t-self.timeSteps,t)
                for sind,sval in enumerate(seq):
                    if sval<0:
                        seq[sind] = 0
                if self.x_data[s,seq[-1],0,0,0] > 0:
                    self.indices.append([s, seq])
        self.numBatches = int(len(self.indices)/self.batch_size)
        self.on_epoch_end()
                                  
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.numBatches
                                  
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_ = self.indexes[index]
        
        # Generate data
        #t0 = time.time()
        X, Y = self.__data_generation(index_)
        #t1 = time.time()
        #print(t1-t0)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.numBatches)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batchN):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,self.timeSteps,self.x_data.shape[2],self.x_data.shape[3],self.x_data.shape[4]))    
        Y = np.zeros((self.batch_size,self.x_data.shape[2],self.x_data.shape[3],self.x_data.shape[4]-1))
        for i,v in enumerate(range(self.batch_size*batchN,self.batch_size*(batchN+1))):
            X[i,:,:,:,:] = self.x_data[self.indices[v][0],self.indices[v][1],:,:,:]
            X[i,:,2,:,0] = X[i,:,0,:,0]
            Y[i,:,:,:] = self.x_data[self.indices[v][0],self.indices[v][1][-1]+1,:,:,1:]
        return X, Y
    
class PatchedModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(PatchedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        
                        saved_correctly = False
                        while not saved_correctly:
                            try:
                                if self.save_weights_only:
                                    self.model.save_weights(filepath, overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                                saved_correctly = True
                            except Exception as error:
                                print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                                sleep(5)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                saved_correctly = False
                while not saved_correctly:
                    try:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        saved_correctly = True
                    except Exception as error:
                        print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                        sleep(5)