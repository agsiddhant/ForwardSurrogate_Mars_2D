import numpy as np
import IPython
from ipywidgets import *
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow
import keras
print(keras.__version__)

from keras import layers

from keras.layers import InputLayer, Reshape, Dense, Dropout, LSTM, ConvLSTM2D, Conv3D, Activation, Add, Lambda, Conv2D, Multiply, Deconvolution2D, Cropping2D, RNN
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.utils import multi_gpu_model 
from keras import regularizers
from keras.callbacks import History 
from keras.constraints import maxnorm
from keras.layers import BatchNormalization
from keras import Input
from keras.regularizers import l2
from keras.engine.topology import Layer

#import kerasncp as kncp
#from kerasncp.tf import LTCCell

from joblib import load, dump
from plstm import PhasedLSTM

from utils_2D_Fwd import DataGenerator_lstm
from utils_2D_Fwd import DataGenerator_conv_lstm
from utils_2D_Fwd import DataGenerator_lstm_full
from utils_2D_Fwd import PatchedModelCheckpoint

from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
    
class k_LSTM:
    def __init__(self, pathSave):
        
        self.pathSave = pathSave
        tf.keras.backend.clear_session()
        
    def lstm(self, _hSize, _NEPOCH, _learnRate, _numSims, _trialNum):
        graph = tf.get_default_graph()
        #l2reg = regularizers.l2(0.000001/self.y_data.shape[0])
        dropRate = 0.05
        def custom_activation(x):
            return x + tf.sin(x) + tf.exp(-tf.pow(x,2)) + tf.pow(x,2) 
        
        activationFunc = 'selu'        
        
        x_inp = Input(shape=(20,848))
        
        for hind, h in enumerate(_hSize[0]):
            if hind+1<len(_hSize[0]):
                returnSeq = True
            else:
                returnSeq = False
            if hind == 0:
                inp = x_inp
            else:
                inp = x
            x = LSTM(h, activation=activationFunc, return_sequences=returnSeq)(inp)
            x = Dropout(dropRate)(x)
        
        for h in _hSize[1]:
            x = Dense(h, activation=activationFunc)(x)
            x = Dropout(dropRate)(x)
        
        x = Dense(840, activation=activationFunc)(x)       
        
        config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        
        name = str(_hSize) + "_ae840_selu_wDrpt_20to1_" + str(_numSims) + "Sims_0p98tr_trial" + str(_trialNum)
        trainRestart = True
        
        paramsTrain = {'trainOrVal': 'train',
                       'batch_size': 16,
                       'numSims': _numSims, 
                       'shuffle': True}
                        
        paramsVal = {'trainOrVal': 'cv',
                       'batch_size': 16,
                       'numSims': 107, 
                       'shuffle': True}
        
        validation_generator = DataGenerator_lstm(self.pathSave, **paramsVal)
        training_generator = DataGenerator_lstm(self.pathSave, **paramsTrain)
        
        def mean_squared_error_pod(y_true, y_pred):
            s_true = tf.linalg.svd(y_true, compute_uv=False)
            s_pred = tf.linalg.svd(y_pred, compute_uv=False)
            return K.mean(K.square(y_pred - y_true), axis=-1) + K.mean(K.square(s_pred - s_true))        
        
        with session.as_default():
            with session.graph.as_default():
                if trainRestart:
                    #seq.compile(loss="mse", optimizer='adam')
                    seq = Model(x_inp, x)
                    seq.compile(loss="mse", optimizer='adam')
                    
                    session = keras.backend.get_session()
                    init = tf.global_variables_initializer()
                    session.run(init)
                else:
                    seq = load_model(self.pathSave + 'TrainedNetworks/LSTM/' + name +'.hdf5')
                    
                seq.summary()
                
                def scheduler(epoch):
                    if epoch<50:
                        return _learnRate
                    elif epoch>=50 and epoch<200:
                        return _learnRate/10
                    else:
                        return _learnRate/100
                
                cp = PatchedModelCheckpoint(self.pathSave + 'TrainedNetworks/LSTM/' + name + '.hdf5', 
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/LSTM/'+name+'.txt', separator=",", append=False)
                
                lstm = seq.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    max_queue_size=10,                 
                    epochs=500,
                    callbacks=[cp, csvlogger, lr_scheduler])
        
        return self 
    
    def lstm_full(self, _hSize, _NEPOCH, _learnRate, _numSims, _trialNum):
        graph = tf.get_default_graph()
        #l2reg = regularizers.l2(0.000001/self.y_data.shape[0])
        dropRate = 0.05
        activationFunc = 'selu'
        
        def custom_activation(x):
            return x + tf.sin(x) + tf.exp(-tf.pow(x,2)) #tf.pow(x,2) #+ 
        
        x_inp = Input(shape=(20,848))
        
        for hind, h in enumerate(_hSize[0]):
            if hind+1<len(_hSize[0]):
                returnSeq = True
            else:
                returnSeq = False
            if hind == 0:
                inp = x_inp
            else:
                inp = x
            x = LSTM(h, activation=activationFunc, return_sequences=returnSeq)(inp)
            x = Dropout(dropRate)(x)
        
        for h in _hSize[1]:
            x = Dense(h, activation=activationFunc)(x)
            x = Dropout(dropRate)(x)
        
        x = Dense(840, activation=activationFunc)(x)       
        x = Reshape((5,7,24))(x)
        
        x = Deconvolution2D(1, (5, 5), strides=(61,57), padding='same', activation=activationFunc)(x)
        x = Cropping2D(cropping=((2, 1), (2, 3)))(x)
        
        config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        
        name = str(_hSize) + "_aeFull_selu_wDrpt_20to1_" + str(_numSims) + "Sims_0p98tr_trial" + str(_trialNum)
        namePrev = str(_hSize) + "_aeFull_selu_wDrpt_20to1_9877Sims_0p98tr_trial" + str(_trialNum)
        trainRestart = False
        
        paramsTrain = {'trainOrVal': 'train',
                       'batch_size': 16,
                       'numSims': _numSims, 
                       'shuffle': True}
                        
        paramsVal = {'trainOrVal': 'cv',
                       'batch_size': 16,
                       'numSims': 107, 
                       'shuffle': True}
        
        validation_generator = DataGenerator_lstm_full(self.pathSave, **paramsVal)
        training_generator = DataGenerator_lstm_full(self.pathSave, **paramsTrain)
        
        with session.as_default():
            with session.graph.as_default():
                if trainRestart:
                    #seq.compile(loss="mse", optimizer='adam')
                    seq = Model(x_inp, x)
                    seq.compile(loss="mse", optimizer='adam')
                    
                    session = keras.backend.get_session()
                    init = tf.global_variables_initializer()
                    session.run(init)
                    
                     #with session2.as_default():
                     #with session2.graph.as_default():

                    #else:
                    ae = ['T_Mars300_sims10k_f5A7s2_c3to24_tanh_l2reg_wlrsc', "conv2d_6", 840, [5,7,24], -11]
                    autoencoder = load_model(self.pathSave + '/TrainedNetworks/autoencoder/' + ae[0] + '.hdf5')
                    #encoded_input = Input(shape=(ae[3][0],ae[3][1],ae[3][2]))
                    #deco = encoded_input
                    x_inp = Input(shape=(20,848))
                    
                    lstm_trained = load_model(self.pathSave + 'TrainedNetworks/LSTM/' + namePrev +'.hdf5')
                    x = lstm_trained(x_inp)
                    x = Reshape((5,7,24))(x)
                    
                    for l in range(ae[4],0):
                        x = autoencoder.layers[l](x)
                    #x = Model(encoded_input, x)
                    seq = Model(x_inp, x)
                    seq.compile(loss="mse", optimizer='adam')    
                    
                else:
                    seq = load_model(self.pathSave + 'TrainedNetworks/LSTM/' + namePrev +'.hdf5')
                    
                
                seq.summary()
                
                def scheduler(epoch):
                    if epoch<20:
                        return _learnRate
                    else:
                        return _learnRate/10
                
                cp = PatchedModelCheckpoint(self.pathSave + 'TrainedNetworks/LSTM/' + name + '.{epoch:02d}-{val_loss:.7f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
                
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/LSTM/'+name+'.txt', separator=",", append=True)
                
                lstm = seq.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=8,
                    verbose=1,
                    max_queue_size=10,                 
                    epochs=500,
                    callbacks=[cp, csvlogger, lr_scheduler])
        
        return self 
    
    def k_ncp(self, _hSize, _NEPOCH, _learnRate, _numSims, _trialNum):
        graph = tf.get_default_graph()
        #l2reg = regularizers.l2(0.000001/self.y_data.shape[0])
        
        wiring = kncp.wirings.FullyConnected(100, 10)  # 8 units, 1 motor neuron
        ltc_cell = LTCCell(wiring) # Create LTC model

        x_inp = Input(shape=(20,848))
        
        for hind, h in enumerate(_hSize[0]):
            if hind+1<len(_hSize[0]):
                returnSeq = True
            else:
                returnSeq = False
            if hind == 0:
                inp = x_inp
            else:
                inp = x
            x = RNN(ltc_cell, return_sequences=returnSeq)(inp)
            x = Dropout(dropRate)(x)
        
        for h in _hSize[1]:
            x = Dense(h, activation=activationFunc)(x)
            x = Dropout(dropRate)(x)
        
        x = Dense(840, activation=activationFunc)(x)       
        
        config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        
        name = str(_hSize) + "_ncp_aeFull_selu_wDrpt_20to1_" + str(_numSims) + "Sims_0p98tr_trial" + str(_trialNum)
        trainRestart = True
        
        paramsTrain = {'trainOrVal': 'train',
                       'batch_size': 16,
                       'numSims': _numSims, 
                       'shuffle': True}
                        
        paramsVal = {'trainOrVal': 'cv',
                       'batch_size': 16,
                       'numSims': 107, 
                       'shuffle': True}
        
        validation_generator = DataGenerator_lstm(self.pathSave, **paramsVal)
        training_generator = DataGenerator_lstm(self.pathSave, **paramsTrain)
        
        with session.as_default():
            with session.graph.as_default():
                if trainRestart:
                    #seq.compile(loss="mse", optimizer='adam')
                    seq = Model(x_inp, x)
                    seq.compile(loss="mse", optimizer='adam')
                    
                    session = keras.backend.get_session()
                    init = tf.global_variables_initializer()
                    session.run(init)                    
                else:
                    seq = load_model(self.pathSave + 'TrainedNetworks/LSTM/' + namePrev +'.hdf5')
                    
                
                seq.summary()
                
                def scheduler(epoch):
                    if epoch<100:
                        return _learnRate
                    elif epoch>=100 and epoch<300:
                        return _learnRate/10
                    else:
                        return _learnRate/100
                
                cp = PatchedModelCheckpoint(self.pathSave + 'TrainedNetworks/LSTM/' + name + '.{epoch:02d}-{val_loss:.7f}.hdf5',
                monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
                
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/LSTM/'+name+'.txt', separator=",", append=True)
                
                lstm = seq.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    max_queue_size=10,                 
                    epochs=1000,
                    callbacks=[cp, csvlogger, lr_scheduler])
        
        return self 
    
    def lstm_wSC(self, _hSize, _NEPOCH, _learnRate):
        graph = tf.get_default_graph()
        #l2reg = regularizers.l2(0.000001/self.y_data.shape[0])
        dropRate = 0.05
        activationFunc = 'selu'
        
        input_vec = Input(shape=(20,848))
        x = LSTM(_hSize[0][0], activation=activationFunc, return_sequences=True)(input_vec)
        
        for h in _hSize[0][1:]:
            x = LSTM(h, activation=activationFunc, return_sequences=False)(x)
            x = Dropout(dropRate)(x)
            
        xl = LSTM(h, activation=activationFunc, return_sequences=False)(input_vec)
        xl = Dropout(dropRate)(xl)
        x = Add()([xl,x])
        
        for h in _hSize[1]:
            x = Dense(h, activation=activationFunc)(x)
            x = Dropout(dropRate)(x)
        
        x = Dense(840, activation=activationFunc)(x)             
        
        config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        
        name = str(_hSize) + '_ae840_selu_wSC_wDrpt_20to1'
        trainRestart = True
        
        paramsTrain = {'trainOrVal': 'train',
                       'batch_size': 16,
                       'numSims': 9080, 
                       'shuffle': True}
                        
        paramsVal = {'trainOrVal': 'cv',
                       'batch_size': 16,
                       'numSims': 501, 
                       'shuffle': True}
        
        validation_generator = DataGenerator_lstm(self.pathSave, **paramsVal)
        training_generator = DataGenerator_lstm(self.pathSave, **paramsTrain)
        
        def mean_squared_error_pod(y_true, y_pred):
            s_true = tf.linalg.svd(y_true, compute_uv=False)
            s_pred = tf.linalg.svd(y_pred, compute_uv=False)
            return K.mean(K.square(y_pred - y_true), axis=-1) + K.mean(K.square(s_pred - s_true))        
        
        with session.as_default():
            with session.graph.as_default():
                if trainRestart:
                    seq = Model(input_vec, x)
                    seq.compile(loss="mse", optimizer='adam')
                else:
                    seq = load_model(self.pathSave + 'TrainedNetworks/LSTM/' + nameAe + '_' + str(_hSize) +'.hdf5')
                    
                seq.summary()
                
                def scheduler(epoch):
                    initial_lrate = _learnRate
                    if epoch < 20:
                        return initial_lrate/10.
                    elif epoch >= 20 and epoch < 200:
                        return initial_lrate/100.
                    else:
                        return initial_lrate/1000.
                session = keras.backend.get_session()
                init = tf.global_variables_initializer()
                session.run(init)
                
                cp = PatchedModelCheckpoint(self.pathSave + 'TrainedNetworks/LSTM/' + name + '.hdf5', 
                monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
                
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/LSTM/'+name+'.txt', separator=",", append=False)
                
                lstm = seq.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    max_queue_size=64,                 
                    epochs=10000,
                    callbacks=[cp, csvlogger, lr_scheduler])
        
        return self
    
class Regression_NN:
    def __init__(self,_pathSave):
        self.pathSave = _pathSave
        
        dictToOpen = "processedDicts/Mars_new/NN/T/Dict_processed_data_MarsNew_2D_NN_T_[12000]sims_1channel_ae840_x_data_0p98tr.txt" 
        with open(_pathSave + dictToOpen, "rb") as myFile:
            self.x_data, self.y_data = load(myFile)
        dictToOpen = "processedDicts/Mars_new/NN/T/Dict_processed_data_MarsNew_2D_NN_T_[12000]sims_1channel_ae840_x_cv_0p98tr.txt" 
        with open(_pathSave + dictToOpen, "rb") as myFile:
            self.x_cv, self.y_cv = load(myFile)
                  
        config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        
        print(self.x_data.shape, self.y_data.shape, self.x_cv.shape, self.y_cv.shape)
        
        tf.keras.backend.clear_session()
        
    def ConvNN(self, _hSize, _NEPOCH, _learnRate, _mpDictTrain={}, _mpDictTest={}):
        
        config = tf.ConfigProto(
        #device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        init_g = tf.global_variables_initializer()
        session.run(init_g)
        GPUS = 1
        name = 'ConvNN_' + str(_hSize) + "_selu_ae840_0p98tr"
        with session.as_default():
            with session.graph.as_default():
                input_img = Input(shape=(self.x_data.shape[1],))  
                #l2reg_gamma = 1e-9
                activFunc = 'selu'
                dropRate = 0.05

                x = {}
                
                x["0"] = Dense(self.y_data.shape[1]*self.y_data.shape[2]*self.y_data.shape[3], activation=activFunc)(input_img)
                x["0"] = Dropout(dropRate)(x["0"])
                x["0"] = Reshape((self.y_data.shape[1],self.y_data.shape[2],self.y_data.shape[3]))(x["0"])
                
                for ind, h in enumerate(_hSize):
                    if ind >0:
                        x[str(ind)] = Conv2D(h, (3,3), strides=(1, 1), padding='same')(x[str(ind-1)])
                    else:
                        x[str(ind)] = Conv2D(h, (3,3), strides=(1, 1), padding='same')(x[str(0)])
                    for i in range(ind):
                        x[str(ind)] = Add()([x[str(ind)],x[str(i)]])
                    x[str(ind)] = Activation(activFunc)(x[str(ind)])
                    x[str(ind)] = Dropout(dropRate)(x[str(ind)])
                    
                x["out"] = Conv2D(self.y_data.shape[3], (3,3), strides=(1, 1), padding='same', activation=activFunc)(x[str(ind)])
                
                def scheduler(epoch):
                    initial_lrate = _learnRate
                    if epoch < 200:
                        return initial_lrate
                    elif epoch >=200 and epoch<500:
                        return initial_lrate/10
                    else:
                        return initial_lrate/100

                NN = Model(input_img, x["out"])
                
                #NN = load_model(self.pathSave + 'TrainedNetworks/NN/' + name +'.hdf5')

                NN.compile(optimizer=keras.optimizers.Adam(), loss="mse") #lr=_learnRate, amsgrad=True
                NN.summary()

                K.get_session().run(tf.global_variables_initializer())

                cp = keras.callbacks.ModelCheckpoint(self.pathSave + 'TrainedNetworks/NN/' + name + '.hdf5', 
                    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/NN/'+name+'.txt', separator=",", append=False)
                
                ae = NN.fit(self.x_data, self.y_data,
                epochs=1000,
                batch_size=np.power(2,4)*GPUS,
                shuffle=True,
                validation_data=(self.x_cv, self.y_cv),
                callbacks=[cp, csvlogger, lr_scheduler])           
        
        return self 
    
    def NN(self, _hSize, _NEPOCH, _learnRate, _mpDictTrain={}, _mpDictTest={}):
        
        config = tf.ConfigProto(
        #device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        init_g = tf.global_variables_initializer()
        session.run(init_g)
        GPUS = 1
        name = 'NN_' + str(_hSize) + "_selu_ae840_0p98tr"
        with session.as_default():
            with session.graph.as_default():
                input_img = Input(shape=(self.x_data.shape[1],))  
                #l2reg_gamma = 1e-9
                activFunc = 'selu'
                dropRate = 0.05
                x = {}
                x["0"] = Dense(_hSize[0], activation=activFunc)(input_img)    
                x["0"] = Dropout(dropRate)(x["0"])
                
                for ind, h in enumerate(_hSize[1:]):
                    if ind >0:
                        x[str(ind)] = Dense(h)(x[str(ind-1)])
                    else:
                        x[str(ind)] = Dense(h)(x[str(0)])
                    for i in range(ind):
                        x[str(ind)] = Add()([x[str(ind)],x[str(i)]])
                    x[str(ind)] = Activation(activFunc)(x[str(ind)])
                    x[str(ind)] = Dropout(dropRate)(x[str(ind)])
                    
                x["out"] = Dense(self.y_data.shape[1]*self.y_data.shape[2]*self.y_data.shape[3], activation=activFunc)(x[str(ind)])
                x["out"] = Reshape((self.y_data.shape[1],self.y_data.shape[2],self.y_data.shape[3]))(x["out"])
                
                def scheduler(epoch):
                    initial_lrate = _learnRate
                    if epoch < 200:
                        return initial_lrate
                    elif epoch >=200 and epoch<500:
                        return initial_lrate/10
                    else:
                        return initial_lrate/100

                NN = Model(input_img, x["out"])
                
                #NN = load_model(self.pathSave + 'TrainedNetworks/NN/' + name +'.hdf5')

                NN.compile(optimizer=keras.optimizers.Adam(), loss="mse") #lr=_learnRate, amsgrad=True
                NN.summary()

                K.get_session().run(tf.global_variables_initializer())

                cp = keras.callbacks.ModelCheckpoint(self.pathSave + 'TrainedNetworks/NN/' + name + '.hdf5', 
                    monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/NN/'+name+'.txt', separator=",", append=False)
                
                ae = NN.fit(self.x_data, self.y_data,
                epochs=1000,
                batch_size=np.power(2,4)*GPUS,
                shuffle=True,
                validation_data=(self.x_cv, self.y_cv),
                callbacks=[cp, csvlogger, lr_scheduler])           
        
        return self
    
class k_convLSTM:
    def __init__(self,pathSave):
        self.pathSave = pathSave                
        tf.keras.backend.clear_session()
        
    def NN(self, _hSize, _NEPOCH, _learnRate, _numSims):
        graph = tf.get_default_graph()
        dropRate = 0.05
        activationFunc = 'selu'
        
        x_inp = Input(shape=(20, 5, 7, 25))
        
        for hind, h in enumerate(_hSize[0]):
            if hind+1<len(_hSize[0]):
                returnSeq = True
            else:
                returnSeq = False
            if hind==0:
                inp = x_inp
            else:
                inp = x
            x = ConvLSTM2D(h,kernel_size=(4,5), strides=(1, 1), padding='same', activation=activationFunc, return_sequences=returnSeq)(inp)
            x = Dropout(dropRate)(x)
        
        for h in _hSize[1][1:-1]:
            x = Conv2D(h,kernel_size=(4,5),strides=(1, 1),padding='same', activation=activationFunc)(x)
            x = Dropout(dropRate)(x)
        
        x = Conv2D(filters=_hSize[1][-1], kernel_size=(4,5), strides=(1, 1), padding="same", activation=activationFunc)(x)       
        
        paramsTrain = {'trainOrVal': 'train',
                       'batch_size': 16,
                       'numSims': _numSims, 
                       'shuffle': True}
                        
        paramsVal = {'trainOrVal': 'cv',
                       'batch_size': 16,
                       'numSims': 107, 
                       'shuffle': True}
        
        validation_generator = DataGenerator_conv_lstm(self.pathSave, **paramsVal)
        training_generator = DataGenerator_conv_lstm(self.pathSave, **paramsTrain)    
        
        name = "ConvLSTM-ae840-" + str(_hSize) + "20to1_" + str(_numSims) + "Sims_0p98tr"
        trainRestart = True
        
        config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        allow_soft_placement=True)

        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        session = tf.Session(config=config)
        
        with session.as_default():
            with session.graph.as_default():
                if trainRestart:
                    seq = Model(x_inp, x)
                    seq.compile(loss="mse", optimizer='adam')
                    session = keras.backend.get_session()
                    init = tf.global_variables_initializer()
                    session.run(init)
                else:
                    seq = load_model(self.pathSave + 'TrainedNetworks/ConvLSTM/' + name +'.hdf5')
                
                seq.summary()
                
                def scheduler(epoch):
                    if epoch < 20:
                        return _learnRate
                    elif epoch >= 20 and epoch <100:
                        return _learnRate/10
                    else:
                        return _learnRate/100
                    
                cp = PatchedModelCheckpoint(self.pathSave + 'TrainedNetworks/ConvLSTM/' + name + '.hdf5', 
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                
                lr_scheduler = LearningRateScheduler(scheduler)
                csvlogger = tf.keras.callbacks.CSVLogger(self.pathSave+'TrainedNetworks/ConvLSTM/'+name+'.txt', separator=",", append=False)
                
                lstm = seq.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    verbose=1,
                    max_queue_size=10,                 
                    epochs=1000,
                    callbacks=[cp, csvlogger, lr_scheduler])
                
        
        return self 
    
class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff,2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.units
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class MDN2Layer(Layer):
    def __init__(self, KMIX, **kwargs):
        self.units = KMIX + KMIX*2 + KMIX*3
        self.KMIX = KMIX
        super(MDN2Layer, self).__init__(**kwargs)
        
    def meshgrid(self, x, y):
        [gx, gy] = np.meshgrid(x, y, indexing='ij')
        gx, gy = np.float64(gx), np.float64(gy)
        grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
        return grid.T.reshape(x.size, y.size, 2)


    def build(self, input_shape):
        self.inp_shape = input_shape
        self.w = self.add_weight(shape=(int(input_shape[1]), self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        super(MDN2Layer, self).build(input_shape)

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
            
        y_pred = tf.reshape(tf.cast(y_pred,"float64"), [-1, self.KMIX*2 + self.KMIX*3 + self.KMIX], name='reshape_ypreds')
        # Split the inputs into paramaters
        out_mu, out_sig, out_pi = tf.split(y_pred, num_or_size_splits=[self.KMIX * 2,
                                                                         self.KMIX * 3,
                                                                         self.KMIX],
                                             axis=-1, name='mdn_coef_split')
        #out_mu = tf.reshape(out_mu, [-1, self.KMIX, 2])
        #out_sig = tf.reshape(out_sig, [-1, self.KMIX, 3])
        
        cat = tfd.Categorical(logits=out_pi)
        component_splits_mu = [2] * self.KMIX
        component_splits_sig = [3] * self.KMIX
        mus = tf.split(tf.cast(out_mu,"float64"), num_or_size_splits=component_splits_mu, axis=1)
        sigs = tf.split(tf.cast(out_sig,"float64"), num_or_size_splits=component_splits_sig, axis=1)
        
        low_diag = tfp.bijectors.FillTriangular(upper=False)
        sigs = [low_diag.forward(scale) for scale in sigs]
        sigs = [tf.linalg.set_diag(scale,tf.exp(tf.linalg.diag_part(scale))) for scale in sigs]
        covs = [tf.matmul(scale,tf.transpose(scale, perm=[0,2,1])) for scale in sigs]        

        x_ = np.linspace(0.,1.,302,dtype=np.float64)
        y_ = np.linspace(0.,1.,394,dtype=np.float64)
        #x_, y_ = tf.meshgrid(x_, y_)
        #coords = np.expand_dims(np.stack((x_, y_), axis=-1),axis=0)
        #self.pos = tf.stack((x_, y_), axis=-1)
        self.pos = self.meshgrid(x_,y_)
        self.pos = tf.reshape(tf.cast(self.pos,"float64"), [-1, 302, 394, 2])
        #for i in range(4):
        #    self.pos = tf.concat((self.pos,self.pos),axis=0)
        constructManual = False
        if constructManual:
            n = 2
            Sigma_det = [tf.linalg.det(cov) for cov in covs]
            Sigma_inv = [tf.linalg.inv(cov) for cov in covs]
            N = [tf.sqrt((2*np.pi)**n * sd) for sd in Sigma_det]
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            res =  0
            for i in range(self.KMIX):
                #fac = tf.einsum('...k,kl,...l->...', self.pos-mus[i], Sigma_inv[i], self.pos-mus[i]) 
                fac = tf.einsum('ijkl,klm->ijkm', self.pos-mus[i], Sigma_inv[i])
                fac = tf.einsum('ijkl,ijkl->ijk', fac, self.pos-mus[i])
                res += tf.exp(-fac / 2) / N[i]
        else:
            coll = [tfd.MultivariateNormalFullCovariance(loc=loc, covariance_matrix=scale) for loc, scale in zip(mus, covs)]
            mixture = tfd.Mixture(cat=cat, components=coll, validate_args=True)
            res = mixture.prob(self.pos)
        res = tf.reshape(res,[-1,302,394,1])
        #print(res.shape)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0],302,394,1)
    
    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.KMIX
        }
        base_config = super(MDN2Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class MDNLayer(Layer):
    def __init__(self, KMIX, **kwargs):
        self.units = 302*KMIX*3
        self.KMIX = KMIX
        super(MDNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(int(input_shape[1]), self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
        super(MDNLayer, self).build(input_shape)

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
            
        y_pred = tf.reshape(tf.cast(y_pred,"float64"), [-1, self.KMIX*3, 302], name='reshape_ypreds')
        # Split the inputs into paramaters
        out_mu, out_sig, out_pi = tf.split(y_pred, num_or_size_splits=3,axis=1, name='mdn_coef_split')
        
        print(out_mu.shape,out_sig.shape,out_pi.shape)
        gms = [tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=out_pi[:,:,i]),
                                   components_distribution=tfd.Normal(loc=out_mu[:,:,i],scale=out_sig[:,:,i])) for i in range(302)]  # And same here.
        pos = tf.reshape(tf.cast(tf.linspace(0.,1.,394), tf.float64), [-1, 394])
        res = tf.convert_to_tensor([gms[i].prob(pos) for i in range(302)])
        res = tf.reshape(res,[-1,302,394,1])
        print(res.shape)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0],302,394,1)
    
    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.KMIX
        }
        base_config = super(MDNLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))