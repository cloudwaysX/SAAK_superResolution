#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 22:12:01 2018

@author: yifang
"""
import numpy as np
import time

import argparse
parser = argparse.ArgumentParser()

########preprocessing parameter##########################
parser.add_argument("--input_size", default = 9, type = int, help = "input size of training patch")
parser.add_argument("--stride", default = 9, type = int, help = "stride when cropping patches from original image")
parser.add_argument("--down_scale", default = 3, type = int, help = "how much downscale you want to get for Low Resolution Image")
########clf parameter##########################
parser.add_argument("--n_cluster_local", default = 10, type=int, help = "how many clusters you want; if not using kmean, should be 1")
parser.add_argument("--n_cluster_global", default = 10, type=int, help = "how many clusters you want; if not using kmean, should be 1")
######################################
parser.add_argument("--mapping_weight_thresh", nargs = '+', default = [0,0], type = float, help = "the weight threshold to drop the HR feature vector")
########Others##########################
parser.add_argument("--printPcaW", action="store_true", help="print out PCA weight?")
parser.add_argument("--printNet", action="store_true", help="print out net structure?")


opt = parser.parse_args()



import readDataset
myDataset = readDataset.DatasetBSD()
myDataset.readBSD_fromMatlab(inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
trainset = myDataset.loadData(dataset = 'training')

def KmeanHelper(input,n_clusters,kmeans = None):  
    from sklearn.cluster import KMeans
    if not kmeans:
        tic = time.clock()
        k_means = KMeans(n_clusters = n_clusters,max_iter=300,n_jobs=6,random_state=0)
        k_means.fit(input)
        kmeans_D = k_means.transform(input)
        print('this kmeans time is: ',time.clock()-tic)
        return k_means,kmeans_D
    else:
        labels = kmeans.predict(input)
        kmeans_D = kmeans.transform(input)
        return labels,kmeans_D
    
def checkCluster(n_clusters,imgBatch_LR,imgBatch_HR,labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20,20))
    
    size = imgBatch_HR.shape[0]
    
    for clusterI in range(n_clusters):
        count = 0  
        for i in range(size):
            if labels[i]==clusterI and np.random.uniform()<0.2:

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
            
def checkComboCluster(imgBatch_LR,imgBatch_HR,labels):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(25,40))
    
    size = imgBatch_HR.shape[0]
    n_clusters = 16
    
    for global_clusterI in range(7):
        for local_clusterI in range(7):
            clusterI = global_clusterI*7+local_clusterI
            count = 0  
            for i in range(size):
                if labels[i,0]==global_clusterI and labels[i,1]==local_clusterI and np.random.uniform()<0.2:
    
                    ax = plt.subplot(n_clusters,12,1+count+12*clusterI)
                    tmp = np.clip(imgBatch_LR[i],0,255)
                    ax.imshow(tmp[0],cmap='gray')
                    ax.set_title("LR({},{})".format(global_clusterI,local_clusterI))
                    
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

def selectCloseSample(kmeansDs,labels,localThresh=0.5,globalTresh=0.5):
    from copy import deepcopy
    kmeansD_local = kmeansDs[1];labels_local = deepcopy(labels[1])
    for local_clusterI in range(opt.n_cluster_local):
        meanD = np.mean(kmeansD_local[:,local_clusterI][labels_local==local_clusterI])  
        closeCenterI = np.squeeze([np.logical_and(labels_local==local_clusterI,kmeansD_local[:,local_clusterI]>meanD*localThresh)])
        labels_local[closeCenterI] = opt.n_cluster_local
        
    kmeansD_global = kmeansDs[0];labels_global = deepcopy(labels[0])
    for global_clusterI in range(opt.n_cluster_global):
        meanD = np.mean(kmeansD_global[:,global_clusterI][labels_global==global_clusterI])  
        closeCenterI = np.squeeze([np.logical_and(labels_global==global_clusterI,kmeansD_global[:,global_clusterI]>meanD*globalTresh)])
        labels_global[closeCenterI] = opt.n_cluster_global
        
    overall_training_label = np.tile(labels_global,3*3)
    overall_training_label = np.stack((overall_training_label,labels_local),axis = 1)
    
    num_usedSample = np.sum((overall_training_label[:,0]+overall_training_label[:,1])!=(opt.n_cluster_local+opt.n_cluster_global))
    print('number of samples can be used (both close to the center for global and local) is {}'.format(num_usedSample))
        
    return overall_training_label
    


####Local Kmeans
trainset_LR_local = createLocalBlocks(trainset_LR)
trainset_HR_local = createLocalBlocks(trainset_HR)

#Get the first Gradient and put into Kmeans
patchNum = trainset_LR_local.shape[0]
trainset_LR_local2 = np.reshape(trainset_LR_local, (int(patchNum), 9))
trainset_LR_local2 = trainset_LR_local2-trainset_LR_local2[:,[4]]
kmeans_local,Ds_local = KmeanHelper(trainset_LR_local2,opt.n_cluster_local)


####global Kmeans
tmp,trainset_LR_global = createGlobalSamples(trainset_LR)
trainset_LR_global = trainset_LR_global - trainset_LR_global[:,[4]]
kmeans_global,Ds_global = KmeanHelper(trainset_LR_global,opt.n_cluster_global)


import calcPCA_weight
import layerExtractor

overall_training_label = selectCloseSample((Ds_global,Ds_local),(kmeans_global.labels_,kmeans_local.labels_),globalTresh=1,localThresh=0.7)

#checkCluster(opt.n_cluster_local,trainset_LR_local,trainset_HR_local,overall_training_label[:,1])
#checkCluster(opt.n_cluster_global,trainset_LR,trainset_HR,overall_training_label[:,0])
#checkComboCluster(trainset_LR_local,trainset_LR_local,overall_training_label)


LR2HR = {}
def lsMapping(clusterIs,LR,HR=None,alreadyCalcMat = False):
    #arrange to shape (patch number, features)
    print('start mapping from LR to HR using LST')
    LR = LR.data.numpy()
    patchNum = LR.shape[0]
    featureNum_LR = LR.shape[1]*LR.shape[2]*LR.shape[3]
    LR_arranged = np.reshape(np.transpose(LR,(0,2,3,1)),(patchNum,featureNum_LR))        
    LR_arranged = np.concatenate((np.ones((LR_arranged.shape[0],1)),LR_arranged),axis=1) #Wx+b = y
    
    if not alreadyCalcMat:
        assert HR is not None, "Need HR value to calculate the mapping matrix"
        # calculate lslq 
        featureNum_HR = HR.shape[1]*HR.shape[2]*HR.shape[3]
        HR_arranged = np.reshape(np.transpose(HR,(0,2,3,1)),(patchNum,featureNum_HR))
            
        results = np.linalg.lstsq(LR_arranged,HR_arranged)
        LR2HR[clusterIs] = results[0]
    elif HR is not None:
        featureNum_HR = HR.shape[1]*HR.shape[2]*HR.shape[3]
        HR_arranged = np.reshape(np.transpose(HR,(0,2,3,1)),(patchNum,featureNum_HR))
        
    
    #do the predicetion
    pred = np.dot(LR_arranged,LR2HR[clusterIs])
    
    MSE = None
#    if not alreadyCalcMat:
#        from sklearn.metrics import mean_squared_error
#        MSE = mean_squared_error(HR_arranged, pred)
#        print('MSE is {}'.format(MSE))

    from sklearn.metrics import mean_squared_error
    MSE = mean_squared_error(HR_arranged, pred)
    print('MSE is {}'.format(MSE))
    
    #calculate score
    from sklearn.metrics import r2_score
    for feaI in range(3*3):
        print('score for feature {} is {}'.format(feaI,r2_score(pred[:,feaI],HR_arranged[:,feaI])))
        
    pred = np.reshape(pred,(patchNum,3,3,1))
    pred = np.transpose(pred,(0,3,1,2))
    
    return pred,MSE

def SVRMapping(clusterIs,LR,HR=None,alreadyCalcMat = False):
    from sklearn.svm import LinearSVR
    #arrange to shape (patch number, features)
    print('start mapping from LR to HR using SVR')
    LR = LR.data.numpy()
    patchNum = LR.shape[0]
    featureNum_LR = LR.shape[1]*LR.shape[2]*LR.shape[3]
    LR_arranged = np.reshape(np.transpose(LR,(0,2,3,1)),(patchNum,featureNum_LR))        
    LR_arranged = np.concatenate((np.ones((LR_arranged.shape[0],1)),LR_arranged),axis=1) #Wx+b = y
    
    if not alreadyCalcMat:
        assert HR is not None, "Need HR value to calculate the mapping matrix"
        # calculate lslq 
        featureNum_HR = HR.shape[1]*HR.shape[2]*HR.shape[3]
        HR_arranged = np.reshape(np.transpose(HR,(0,2,3,1)),(patchNum,featureNum_HR))
        LR2HR[clusterIs] = {}
        for feaI in range(featureNum_HR):
            clf = LinearSVR(random_state = 0,epsilon=0.2,loss = 'epsilon_insensitive')
            clf.fit(LR_arranged,HR_arranged[:,feaI])
            LR2HR[clusterIs][feaI] = clf
    else:
        featureNum_HR = len(LR2HR[clusterIs])    
    
    #do the predicetion
    pred = np.zeros((LR_arranged.shape[0],featureNum_HR))            
    for feaI in range(featureNum_HR):
        pred[:,feaI] = LR2HR[clusterIs][feaI].predict(LR_arranged)
    
    MSE = None
    if not alreadyCalcMat:
        from sklearn.metrics import mean_squared_error
        MSE = mean_squared_error(HR_arranged, pred)
        print('MSE is {}'.format(MSE))
    
    #calculate score
    from sklearn.metrics import r2_score
    for feaI in range(3*3):
        print('score for feature {} is {}'.format(feaI,r2_score(pred[:,feaI],HR_arranged[:,feaI])))
        
    pred = np.reshape(pred,(patchNum,3,3,1))
    pred = np.transpose(pred,(0,3,1,2))
    
    return pred,MSE


import torch
#for global_clusterI in range(opt.n_cluster_global):
#    for local_clusterI in range(opt.n_cluster_local):
for global_clusterI in range(6):
    for local_clusterI in range(6):
        cur_cluster_features_LR_local = trainset_LR_local[np.logical_and(overall_training_label[:,0]==global_clusterI,overall_training_label[:,1]==local_clusterI)]
        if cur_cluster_features_LR_local.shape[0] <= 9: continue
        print(global_clusterI,local_clusterI)
        print(cur_cluster_features_LR_local.shape[0])
#        params_afterClassify_LR_local = {"n_keptComponent":opt.mapping_weight_thresh,"zoom factor":{'1':3},'cluster index':(global_clusterI,local_clusterI)}  
#        cur_cluster_features_LR_local= calcPCA_weight.calcW(cur_cluster_features_LR_local,params_afterClassify_LR_local,isbeforeCalssify=False,in_out_layers=['L0','L1'],\
#                                                       mode = 'LR_scale_{}_interpo'.format(opt.down_scale),printPCAratio = opt.printPcaW)
        params_afterClassify_LR_local = {"n_keptComponent":opt.mapping_weight_thresh,"zoom factor":{'1':3},'cluster index':(global_clusterI,local_clusterI)}  
        cur_cluster_features_LR_local = layerExtractor.forward(cur_cluster_features_LR_local,params_afterClassify_LR_local,isbeforeCalssify=False,\
                                                          mode='LR_scale_{}_interpo'.format(opt.down_scale),in_out_layers =['L0','L1'], savePatch=False,printNet = opt.printNet)


        cur_cluster_features_LR_global = np.tile(trainset_LR,(3*3,1,1,1))
        cur_cluster_features_LR_global = cur_cluster_features_LR_global[np.logical_and(overall_training_label[:,0]==global_clusterI,overall_training_label[:,1]==local_clusterI)]
#        params_afterClassify_LR_global = {"n_keptComponent":opt.mapping_weight_thresh,"zoom factor":{'1':3,'2':3},'cluster index':(global_clusterI,local_clusterI)}  
#        cur_cluster_features_LR_global= calcPCA_weight.calcW(cur_cluster_features_LR_global,params_afterClassify_LR_global,isbeforeCalssify=False,in_out_layers=['L0','L2'],\
#                                                       mode = 'LR_scale_{}_interpo'.format(opt.down_scale),printPCAratio = opt.printPcaW)
        params_afterClassify_LR_global = {"n_keptComponent":opt.mapping_weight_thresh,"zoom factor":{'1':3,'2':3},'cluster index':(global_clusterI,local_clusterI)}  
        cur_cluster_features_LR_global = layerExtractor.forward(cur_cluster_features_LR_global,params_afterClassify_LR_global,isbeforeCalssify=False,\
                                                          mode='LR_scale_{}_interpo'.format(opt.down_scale),in_out_layers =['L0','L2'], savePatch=False,printNet = opt.printNet)

#
        print(cur_cluster_features_LR_global.size())
        print(cur_cluster_features_LR_local.size())
        cur_cluster_features_LR = torch.cat((cur_cluster_features_LR_global,cur_cluster_features_LR_local),dim=1)


#        cur_cluster_features_LR = cur_cluster_features_LR_local


        
        cur_cluster_features_HR = trainset_HR_local[np.logical_and(overall_training_label[:,0]==global_clusterI,overall_training_label[:,1]==local_clusterI)]
        cur_cluster_features_HR_pred,_ = lsMapping((global_clusterI,local_clusterI),cur_cluster_features_LR,cur_cluster_features_HR)
#        cur_cluster_features_HR_pred,_ = SVRMapping((global_clusterI,local_clusterI),cur_cluster_features_LR,cur_cluster_features_HR)







myDataset.readBSD_fromMatlab(dataset = 'testing',inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
testset = myDataset.loadData(dataset = 'testing')
testset_LR = testset['LR_scale_{}_interpo'.format(opt.down_scale)]
testset_HR = testset['HR']

####Local Kmeans
testset_LR_local = createLocalBlocks(testset_LR)
testset_HR_local = createLocalBlocks(testset_HR)

#Get the first Gradient and put into Kmeans
patchNum = testset_LR_local.shape[0]
testset_LR_local2 = np.reshape(testset_LR_local, (int(patchNum), 9))
testset_LR_local2 = testset_LR_local2-testset_LR_local2[:,[4]]
labels_local,Ds_local_test = KmeanHelper(testset_LR_local2,opt.n_cluster_local,kmeans_local)


####global Kmeans
tmp,testset_LR_global = createGlobalSamples(testset_LR)
testset_LR_global = testset_LR_global - testset_LR_global[:,[4]]
labels_global,Ds_global_test = KmeanHelper(testset_LR_global,opt.n_cluster_global,kmeans_global)


overall_testing_label = selectCloseSample((Ds_global_test,Ds_local_test),(labels_global,labels_local),globalTresh=1,localThresh=0.7)

#checkCluster(opt.n_cluster_local,testset_LR_local,testset_HR_local,overall_testing_label[:,1])
#checkCluster(opt.n_cluster_global,testset_LR,testset_HR,overall_testing_label[:,0])
#checkComboCluster(testset_LR_local,testset_LR_local,overall_testing_label)

import layerExtractor
for global_clusterI in range(6):
    for local_clusterI in range(6):
        cur_cluster_features_LR_local = testset_LR_local[np.logical_and(overall_testing_label[:,0]==global_clusterI,overall_testing_label[:,1]==local_clusterI)]
        if cur_cluster_features_LR_local.shape[0] == 0 or (global_clusterI,local_clusterI) not in LR2HR: continue
        print(global_clusterI,local_clusterI)
        print(cur_cluster_features_LR_local.shape[0])
        params_afterClassify_LR_local = {"n_keptComponent":opt.mapping_weight_thresh,"zoom factor":{'1':3},'cluster index':(global_clusterI,local_clusterI)}  
        cur_cluster_features_LR_local = layerExtractor.forward(cur_cluster_features_LR_local,params_afterClassify_LR_local,isbeforeCalssify=False,\
                                                          mode='LR_scale_{}_interpo'.format(opt.down_scale),in_out_layers =['L0','L1'], savePatch=False,printNet = opt.printNet)
        
        cur_cluster_features_LR_global = np.tile(testset_LR,(3*3,1,1,1))
        cur_cluster_features_LR_global = cur_cluster_features_LR_global[np.logical_and(overall_testing_label[:,0]==global_clusterI,overall_testing_label[:,1]==local_clusterI)]
        params_afterClassify_LR_global = {"n_keptComponent":opt.mapping_weight_thresh,"zoom factor":{'1':3,'2':3},'cluster index':(global_clusterI,local_clusterI)}  
        cur_cluster_features_LR_global = layerExtractor.forward(cur_cluster_features_LR_global,params_afterClassify_LR_global,isbeforeCalssify=False,\
                                                          mode='LR_scale_{}_interpo'.format(opt.down_scale),in_out_layers =['L0','L2'], savePatch=False,printNet = opt.printNet)

        cur_cluster_features_LR = torch.cat((cur_cluster_features_LR_global,cur_cluster_features_LR_local),dim=1)
        
        cur_cluster_features_HR = testset_HR_local[np.logical_and(overall_testing_label[:,0]==global_clusterI,overall_testing_label[:,1]==local_clusterI)]
        if cur_cluster_features_LR.size()[1] != LR2HR[(global_clusterI,local_clusterI)].shape[0]-1: continue
        cur_cluster_features_HR_pred,_ = lsMapping((global_clusterI,local_clusterI),cur_cluster_features_LR,cur_cluster_features_HR,alreadyCalcMat=True)
#        cur_cluster_features_HR_pred,_ = SVRMapping((global_clusterI,local_clusterI),cur_cluster_features_LR,cur_cluster_features_HR)