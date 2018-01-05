#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:12:01 2018

@author: yifang
"""
import numpy as np

import argparse
parser = argparse.ArgumentParser()

########preprocessing parameter##########################
parser.add_argument("--input_size", default = 9, type = int, help = "input size of training patch")
parser.add_argument("--stride", default = 9, type = int, help = "stride when cropping patches from original image")
parser.add_argument("--down_scale", default = 3, type = int, help = "how much downscale you want to get for Low Resolution Image")
########clf parameter##########################
parser.add_argument("--n_cluster1", default = 10, type=int, help = "how many clusters you want; if not using kmean, should be 1")

opt = parser.parse_args()



import readDataset
myDataset = readDataset.DatasetBSD()
myDataset.readBSD_fromMatlab(inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
trainset = myDataset.loadData(dataset = 'training')

def KmeanHelper(input):    
    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters = opt.n_cluster1,max_iter=300,n_jobs=6,random_state=0)
    k_means.fit(input)
    kmeans_D = k_means.transform(input)
    return k_means,kmeans_D

def showImage(imgBatch_LR,imgBatch_HR,index=0):
    import matplotlib.pyplot as plt
    
    ax = plt.subplot("141")
    tmp = np.clip(imgBatch_HR[index],0,255)
    ax.imshow(tmp[0],cmap='gray')
    ax.set_title("HR {}".format(index))

    ax = plt.subplot("142")
    tmp = np.clip(imgBatch_LR[index],0,255)
    ax.imshow(tmp[0],cmap='gray')
    ax.set_title("Input(bicubic)")
    
    ax = plt.subplot("143")
    tmp = np.clip(imgBatch_HR[index]-imgBatch_LR[index],0,255)
    ax.imshow(tmp[0],cmap='gray')
    ax.set_title("diff")
    
    plt.show()
    
def checkCluster(clusterI,imgBatch_LR,imgBatch_HR,labels,kmeansD):
    from sklearn.preprocessing import scale
    
    size = imgBatch_HR.shape[0]
    count = 0
    
    meanD = np.mean(kmeansD[:,clusterI][labels==clusterI])
    for i in range(size):
        if labels[i]==clusterI and (kmeansD[:,clusterI][i]<=meanD*0.7):
            print('distance is {}'.format(kmeansD[:,clusterI][i]))
            print('var for distance is {}'.format(scale(np.var(kmeansD,axis=1))[i]))
            showImage(imgBatch_LR,imgBatch_HR,index=i)
            count +=1
        if count == 5: break
    
def createLocalBlocks(input):
    from skimage.util.shape import view_as_windows
    import numpy as np
    
    assert len(input.shape) == 4, 'need input dimension = 4'
    trainset_local = np.transpose(input,(2,3,0,1))
    trainset_local = np.reshape(trainset_local,(9,9,trainset_local.shape[2]*trainset_local.shape[3]))
    trainset_local = view_as_windows(trainset_local, window_shape=(3,3,1),step=(3,3,1))
    patchNum = trainset_local.shape[0]*trainset_local.shape[1]*trainset_local.shape[2]
    trainset_local = np.reshape(trainset_local, (int(patchNum), 9))
    #Get the first Gradient
    trainset_local = trainset_local-trainset_local[:,[4]]
    
    return trainset_local

def createGlobalSamples(input):
    from skimage.transform import downscale_local_mean
    trainset_global = downscale_local_mean(input,(1,1,3,3))
    trainset_global = np.reshape(trainset_global,(trainset_global.shape[0]*trainset_global.shape[1],trainset_global.shape[2]*trainset_global.shape[3]))
    return trainset_global

trainset_LR = trainset['LR_scale_{}_interpo'.format(opt.down_scale)][:1000]
trainset_HR = trainset['HR'][:1000]



#kmeans,Ds = KmeanHelper(np.concatenate((trainset_LR,trainset_HR),axis=1))
#trainset_HR_local = np.reshape(trainset_HR,(trainset_HR.shape[0],1,3,3))
#trainset_LR_local = np.reshape(trainset_LR,(trainset_HR.shape[0],1,3,3))
#checkCluster(1,trainset_LR_local,trainset_HR_local,kmeans.labels_,kmeansD=Ds)



trainset_LR_global = createGlobalSamples(trainset_LR)
kmeans_global,Ds_global = KmeanHelper(trainset_LR_global)
checkCluster(1,trainset_LR,trainset_HR,kmeans_global.labels_,kmeansD=Ds_global)
