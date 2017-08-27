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



def import_data():
    dataset = pd.read_csv('../data/DP.csv',usecols=[27],engine='python',skipfooter=None)
    return dataset
def mean_absolute_percentage_error(y_true, y_pred): 

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def Data_Kmeans_3_Clusters():
    data = import_data()
    data= np.array(data)
    data= data.reshape(-1,24)

    daydata= df(data)
    daydata.index= pd.date_range('2014-1-1', periods=365, freq='D')


    clusters = cluster.KMeans(n_clusters=3).fit_predict(daydata)
    pca= PCA(n_components=3)
    pca.fit(daydata)
    data_pca =  pca.transform(daydata)

    clusters[[45,46,47]]= clusters[0]
    clusters[[84,85,86]]= clusters[87]
    # pdClusters= pd.DataFrame(clusters,index=daydata.index)
    print(clusters)
    fig = plt.figure(1, figsize=(12, 9))

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=45, azim=135)

    ax.scatter(data_pca[:,0],data_pca[:,1],data_pca[:,2],cmap=plt.cm.spectral,c=clusters)

    plt.show()

    daydata['cluster']=clusters
    return daydata

def prepareInputs(daydata, season, UsedInputs):
    nbrInputs= 0
    
    previousHours = UsedInputs[0]
    previousDay = UsedInputs[1]
    previousWeek = UsedInputs[2]
    temp = UsedInputs[3]
    tempMax = UsedInputs[4]
    tempMin = UsedInputs[5]
    dayIndicator = UsedInputs[6]
    
    if previousHours == True: nbrInputs= nbrInputs+ 1
    if previousDay == True: nbrInputs= nbrInputs+1
    if previousWeek == True : nbrInputs= nbrInputs+ 1
    if temp == True: nbrInputs= nbrInputs+1
    if tempMax == True: nbrInputs= nbrInputs+1
    if tempMin == True: nbrInputs= nbrInputs+1
    if dayIndicator== True: nbrInputs= nbrInputs+7

    hourclusters= np.empty([(daydata.index.size*24),1])

    hourdataindex= pd.DataFrame(index=pd.date_range('2014-1-8 00:00:00', periods=(365)*24, freq='H'))

    for x in range(0,daydata.index.size):
        for y in range(0,24):
            hourclusters[(x * 24) + y,0] = daydata.iloc[x,24]
    hourclusters.size

    tempAlgiers=  pd.read_csv('../data/tempAlgiers.csv')
    tempA= tempAlgiers.loc[:,'Hour_1':'Hour_24']
    tempnp= np.array(tempA)
    tempnp= tempnp.reshape(-1,1)
    tempdata= pd.DataFrame(tempnp)

    tempmax= tempAlgiers.loc[:,'Tmax']
    tempmin= tempAlgiers.loc[:,'Tmin']




    tempmx= np.random.random([tempmax.size*24,1])
    tempmn= np.random.random([tempmin.size*24,1])



    for x in range(0,tempmax.size):
        for y in range(0,24):
            tempmx[(x * 24) + y,0] = tempmax.iloc[x]

    for x in range(0,tempmin.size):
        for y in range(0,24):
            tempmn[(x * 24) + y,0] = tempmin.iloc[x]
        

    samples = daydata.index.size*24
    daydata2= daydata.copy()
    del(daydata2['cluster'])

    data= pd.DataFrame(np.array(daydata2).reshape(-1,1))

    maxcons= data.values.max()
    mincons= data.values.min()

    maxtemp= np.max(tempdata.values)
    mintemp= tempdata.values.min()

    maxtempmax= np.max(tempmx)
    mintempmax= np.min(tempmx)

    maxtempmin= np.max(tempmn)
    mintempmin= np.min(tempmn)

    sigxx= np.empty((samples - 168 , nbrInputs))
    sigyy= np.empty((samples - 168 , 1))

    i= 0
    for x in list(range(168,samples)):
        i=0
        if previousHours == True: 
            sigxx[x - 168 , i] = (data.iloc[x - 1 , 0])/(2*maxcons)
            i= i+ 1
        if previousDay == True: 
            sigxx[x - 168 , i] = (data.iloc[x - 24 , 0])/(2*maxcons)
            i= i+1
        if previousWeek == True : 
            sigxx[x - 168 , i] = (data.iloc[x - 168 , 0])/(2*maxcons)
            i= i+ 1
        if temp == True: 
            sigxx[x - 168 , i] = (tempdata.iloc[x])/(2*maxtemp)
            i= i+1
        if tempMax == True: 
            sigxx[x - 168 , i] = (tempmx[x])/(2*maxtempmax)
            i= i+1
        if tempMin == True: 
            sigxx[x - 168 , i] = (tempmn[x])/(2*maxtempmin)
            i= i+1
        if dayIndicator == True:
            ind=0
            for y in range(0,7):
                sigxx[x - 168 , i+ind]= 0
                ind = ind + 1
            sigxx[x - 168 , i+pd.datetime.weekday(hourdataindex.index[x])]=1

    
    for x in list(range(168,samples)):
        sigyy[x - 168 , 0]= (data.iloc[x , 0])/(2*maxcons)

    sigmoidxx= df(sigxx.copy())
    sigmoidyy= df(sigyy.copy())

    sigmoidxx.index= pd.date_range('2014-1-8 00:00:00', periods=(365-7)*24, freq='H')
    sigmoidyy.index= pd.date_range('2014-1-8 00:00:00', periods=(365-7)*24, freq='H')

    sigmoidxx['cluster'] = hourclusters[168:]
    sigmoidyy['cluster'] = hourclusters[168:]
    dfhourclusters = df(hourclusters)
    
    temp1= sigmoidyy[sigmoidyy.cluster==0]
    temp2= sigmoidyy[sigmoidyy.cluster==1]
    temp3= sigmoidyy[sigmoidyy.cluster==2]

    if season == 'summer':
        if temp1.index[0] == pd.datetime(2014,4,9,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==0].copy()
        elif temp2.index[0] == pd.datetime(2014,4,9,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==1].copy()
        elif temp3.index[0] == pd.datetime(2014,4,9,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==2].copy()
    elif season == 'winter':
        if temp1.index[0] == pd.datetime(2014,1,8,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==0].copy()
        elif temp2.index[0] == pd.datetime(2014,1,8,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==1].copy()
        elif temp3.index[0] == pd.datetime(2014,1,8,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==2].copy()
    elif season == 'spring and autumn':
        if temp1.index[0] == pd.datetime(2014,3,18,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==0].copy()
        elif temp2.index[0] == pd.datetime(2014,3,18,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==1].copy()
        elif temp3.index[0] == pd.datetime(2014,3,18,0,0,0): SigmoidInputs =sigmoidxx[sigmoidxx.cluster==2].copy()
    
    SigmoidOutputs= sigmoidyy[sigmoidyy.cluster==SigmoidInputs.loc[SigmoidInputs.index[0],'cluster']]
    del(SigmoidInputs['cluster'],SigmoidOutputs['cluster'])
    
    learningoutputs = pd.DataFrame(SigmoidOutputs.iloc[:int(SigmoidOutputs.size-168)].values.copy(),
                            index=SigmoidOutputs.iloc[:int(SigmoidOutputs.size-168)].index)
    testoutputs = pd.DataFrame(SigmoidOutputs.iloc[int(SigmoidOutputs.size-168):].values.copy(),
                            index=SigmoidOutputs.iloc[int(SigmoidOutputs.size-168):].index)

    learninginputs = pd.DataFrame(SigmoidInputs.iloc[:int(SigmoidOutputs.size-168)].values.copy(),
                            index=SigmoidOutputs.iloc[:int(SigmoidOutputs.size-168)].index)
    testinputs = pd.DataFrame(SigmoidInputs.iloc[int(SigmoidOutputs.size-168):].values.copy(),
                            index=SigmoidOutputs.iloc[int(SigmoidOutputs.size-168):].index)

    print('-------Input preparation process complet-------')
    return learninginputs, learningoutputs, testinputs, testoutputs, nbrInputs

