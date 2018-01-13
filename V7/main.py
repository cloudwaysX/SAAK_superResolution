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
parser.add_argument("--n_cluster_local", default = 10, type=int, help = "how many clusters you want; if not using kmean, should be 1")
parser.add_argument("--n_cluster_global", default = 8, type=int, help = "how many clusters you want; if not using kmean, should be 1")


opt = parser.parse_args()



import readDataset
myDataset = readDataset.DatasetBSD()
myDataset.readBSD_fromMatlab(inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
trainset = myDataset.loadData(dataset = 'training')

def KmeanHelper(input,n_clusters):    
    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters = n_clusters,max_iter=300,n_jobs=6,random_state=0)
    k_means.fit(input)
    kmeans_D = k_means.transform(input)
    return k_means,kmeans_D
    
def checkCluster(n_clusters,imgBatch_LR,imgBatch_HR,labels,kmeansD):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,35))
    
    size = imgBatch_HR.shape[0]
    
    for clusterI in range(n_clusters):
        count = 0
        meanD = np.mean(kmeansD[:,clusterI][labels==clusterI])    
        for i in range(size):
            if labels[i]==clusterI and (kmeansD[:,clusterI][i]<=meanD*0.6):
            
                ax = plt.subplot(n_clusters,12,1+count+12*clusterI)
                tmp = np.clip(imgBatch_LR[i],0,255)
                ax.imshow(tmp[0],cmap='gray')
                ax.set_title("LR")
                
                ax = plt.subplot(n_clusters,12,count+7+12*clusterI)
                tmp = np.clip(imgBatch_HR[i],0,255)
                ax.imshow(tmp[0],cmap='gray')
                ax.set_title("HR {}".format(i))
                
                count +=1
            if count == 6: break
            
def createLocalBlocks(input):
    from skimage.util.shape import view_as_windows
    import numpy as np
    
    assert len(input.shape) == 4, 'need input dimension = 4'
    local = np.transpose(input,(2,3,0,1))
    local = np.reshape(local,(9,9,local.shape[2]*local.shape[3]))
    local = view_as_windows(local, window_shape=(3,3,1),step=(3,3,1))
    patchNum = local.shape[0]*local.shape[1]*local.shape[2]
    local = np.reshape(local,(patchNum,1,3,3))
    
    return local

def createGlobalSamples(input):
    from skimage.transform import downscale_local_mean
    myGlobal = downscale_local_mean(input,(1,1,3,3))
    tmp = myGlobal
    myGlobal = np.reshape(myGlobal,(myGlobal.shape[0]*myGlobal.shape[1],myGlobal.shape[2]*myGlobal.shape[3]))
#    myGlobal = scale(myGlobal)
    return tmp,myGlobal

trainset_LR = trainset['LR_scale_{}_interpo'.format(opt.down_scale)]
trainset_HR = trainset['HR']

trainset_LR_local = createLocalBlocks(trainset_LR)
trainset_HR_local = createLocalBlocks(trainset_HR)

#Get the first Gradient and put into Kmeans
patchNum = trainset_LR_local.shape[0]
trainset_LR_local2 = np.reshape(trainset_LR_local, (int(patchNum), 9))
trainset_LR_local2 = trainset_LR_local2-trainset_LR_local2[:,[4]]
kmeans_local,Ds_local = KmeanHelper(trainset_LR_local2,opt.n_cluster_local)
checkCluster(opt.n_cluster_local,trainset_LR_local,trainset_HR_local,kmeans_local.labels_,kmeansD=Ds_local)

tmp,trainset_LR_global = createGlobalSamples(trainset_LR)
#trainset_LR_global[:,4]=trainset_LR_global[:,4]*2; 
trainset_LR_global = trainset_LR_global - trainset_LR_global[:,[4]]
#from sklearn.preprocessing import scale
#trainset_LR_global = scale(trainset_LR_global)
kmeans_global,Ds_global = KmeanHelper(trainset_LR_global,opt.n_cluster_global)
checkCluster(opt.n_cluster_global,trainset_LR,trainset_HR,kmeans_global.labels_,kmeansD=Ds_global)
#checkCluster(opt.n_cluster_global,tmp,trainset_HR,kmeans_global.labels_,kmeansD=Ds_global)
