import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster, datasets
from pandas import DataFrame as df
from IPython.display import clear_output

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.core import TimeDistributedDense, Activation, Dropout  
from keras.optimizers import RMSprop
from keras.layers import Embedding, RepeatVector
from keras.layers import LSTM
from keras.layers.recurrent import GRU
from keras.optimizers import SGD

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from gp_extras.kernels import ManifoldKernel
    
def create_TrainModel(n_features,RBFInputs=None,alpha=None,whiteKernel=None,n_restarts=None,inputx1,outputx1):
    
    if alpha is None: alpha = 0
    if RBFInputs is None: RBFInputs = 3
    if n_restarts is None: n_restarts = 5
    if whiteKernel is None: whiteKernel = False
        
    if whiteKernel== True:
        kernel = C(1.0, (1e-10, 100)) * RBF([RBFInputs] * n_features,[(0.1, 100.0)] * n_features) \
        + WhiteKernel(1e-3, (1e-10, 1e-1))
    else :
        kernel = C(1.0, (1e-10, 100)) * RBF([RBFInputs] * n_features,[(0.1, 100.0)] * n_features)
        
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts)
    
    # Gaussian Process with Manifold kernel (using an isotropic RBF kernel on
    # manifold for learning the target function)
    # Use an MLP with one hidden-layer for the mapping from data space to manifold
    
    gp.fit(inputx1, outputx1)

    
    return gp

def getForecastingResults(learninginputs,learningoutputs,testinputs,testoutputs,Model, nbrInputs, IsShift,previoushour,season):
    
    predictionPeriod = learninginputs.index.size-168
    temp=learninginputs.iloc[:predictionPeriod].resample('D').sum()
    temp= temp.dropna()

    LearningForecastingError=pd.DataFrame(np.random.randn((predictionPeriod)/24-1, 1))

    learningInputs= learninginputs.iloc[:learningoutputs.index.size-168].copy()
    learningOutputs= learningoutputs.iloc[:learningoutputs.index.size-168].copy()
    LearningInputs_24= learninginputs.iloc[:learninginputs.index.size-168].copy()

    LearningHourlyError=pd.DataFrame(np.random.randn((predictionPeriod-24), 1))

    print('===========================================================================')
    print('\t \t learning subset results')
    print('===========================================================================')

    for y in range(0,predictionPeriod/24-1):

        
        
            
        dailyError = 0

        for x in range(0,24):

            ####    hourly forecasting (get hourly error) 
            
            if IsShift == True:
                hourlyForcast= Model.predict(learningInputs.iloc[x + (y * 24) +1].reshape(-1,nbrInputs))
            else :
                hourlyForcast= Model.predict(learningInputs.iloc[x + (y * 24)].reshape(-1,nbrInputs))

            HourlyError = np.abs(((learningOutputs.iloc[x + (y * 24),0] - hourlyForcast[0]) / 
                                              learningOutputs.iloc[x + (y * 24),0]))
            LearningHourlyError.iloc[x + (y * 24)]= HourlyError
            
            
            ####    make forecasting in a way we get daily error (daily forecasting)
            
            if IsShift == True:
                Testforecast=Model.predict(LearningInputs_24.iloc[x + (y * 24) + 1].reshape(-1,nbrInputs))
                forecast=Model.predict(learningInputs.iloc[x + (y * 24) + 1].reshape(-1,nbrInputs))
            else :
                Testforecast=Model.predict(LearningInputs_24.iloc[x + (y * 24)].reshape(-1,nbrInputs))
                forecast=Model.predict(learningInputs.iloc[x + (y * 24)].reshape(-1,nbrInputs))
                
            dailyError = dailyError + np.abs(((learningOutputs.iloc[x + (y * 24),0] - Testforecast[0]) / 
                                              learningOutputs.iloc[x + (y * 24),0]))
            if previoushour == True: 
                if IsShift == True: 
                    if x + 2 < 24: LearningInputs_24.iloc[(x + (y * 24)) + 2,0] = forecast[0]
                else :
                    if x + 1 < 24: LearningInputs_24.iloc[(x + (y * 24)) + 1,0] = forecast[0]


        dailyError = (dailyError / 24) * 100

        LearningForecastingError.iloc[y,0] = dailyError

    print(LearningHourlyError)
    HourlyLearningMAPE = LearningHourlyError.mean()
    print('Hourly Learning mean absolute percentage error of %d samples : %f ' %(LearningHourlyError.shape[0] , HourlyLearningMAPE[0]))


    LearningMAPE = LearningForecastingError.mean()
    print('Daily Learning mean absolute percentage error: %f ' %LearningMAPE[0])

    fig = plt.figure(figsize=(12,3))
    plt.plot(LearningForecastingError,label='Error')

    if season == 'winter':
        dstart='2014-12-25 00:00:00'
        dend = '2014-12-30 23:00:00'
        dend2= '2014-12-31 00:00:00'
    elif season == 'spring and autumn':
        dstart='2014-11-28 00:00:00'
        dend = '2014-12-03 23:00:00'
        dend2 = '2014-12-04 00:00:00'
    elif season == 'summer':
        dstart='2014-10-29 00:00:00'
        dend = '2014-11-02 23:00:00'
        dend2 = '2014-11-03 00:00:00'
        
    testperiod=len(pd.date_range(dstart,dend,freq='H'))


    forecasting_history=pd.DataFrame(np.random.randn(testperiod, 2),columns=['real', 'forecast']
                                     ,index= pd.date_range(dstart,dend,freq='H'))

    validationInputs= testinputs.loc[dstart:dend2,:].copy()
    validationOutputs= testoutputs.loc[dstart:dend,:].copy()

    for x in pd.date_range(dstart,dend,freq='H'):
        
        if IsShift == True:
            forecast=Model.predict(validationInputs.loc[x+ pd.DateOffset(hours=1)].reshape(-1,nbrInputs))
        else :
            forecast=Model.predict(validationInputs.loc[x+ pd.DateOffset(hours=0)].reshape(-1,nbrInputs))
        
        forecasting_history.loc[x,'forecast']= forecast[0][0]
        forecasting_history.loc[x,'real']= testoutputs.loc[x,0]
        
        if previoushour == True: 
            if IsShift == True:
                if (pd.Timedelta(x-(x + pd.DateOffset(hours=2))).seconds/3600) + 1 < 168 - 1: 
                    validationInputs.loc[x + pd.DateOffset(hours=2),0] = forecast[0]
            else :
                if (pd.Timedelta(x-(x + pd.DateOffset(hours=1))).seconds/3600) + 1 < 168 - 1: 
                    validationInputs.loc[x + pd.DateOffset(hours=1),0] = forecast[0]



    testRMSS = rmse(forecasting_history.iloc[:,0], forecasting_history.iloc[:,1])
    testMAPE= mean_absolute_percentage_error(forecasting_history.iloc[:,0], forecasting_history.iloc[:,1])
    
    print('===========================================================================')
    print('\t \t Test subset results')
    print('===========================================================================')


    print('Test mean squared error: %f' %testRMSS)
    print('Test mean absolute percentage error: %f ' %testMAPE)



    fig = plt.figure(figsize=(12,4))
    plt.plot(forecasting_history[:testperiod], label=['real','forecast'])
    plt.legend(labels=['real','forecast'],loc= 'best')
    plt.show()
    
    return LearningMAPE, testMAPE, HourlyLearningMAPE

