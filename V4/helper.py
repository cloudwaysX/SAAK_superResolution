#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:07:36 2017

@author: yifang
"""

import readMNIST
import layerExtractor
import calcPCA_weight
import numpy as np

myDataset = readMNIST.DatasetMNIST()
#myDataset.writeTest()





myDataset.readMNIST()
trainset = myDataset.loadData(dataset = 'training')
myDataset.readMNIST(dataset='testing')
testset = myDataset.loadData(dataset = 'testing')

needCalcWeight = True
testInverse = False

if needCalcWeight:
#    calcPCA_weight.calcW(trainset,mode='HR',folder = 'MNIST_weight_0_1e-4_0',weight_thresholds = [0,1e-4,0])
    calcPCA_weight.calcW(trainset,mode='LR_scale_6_interpo',folder = 'MNIST_weight_0_0_1e-7',weight_thresholds = [0,0,1e-7])

#forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
#                                      ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',weight_thresholds = [0,0,0])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'MNIST_weight',weight_thresholds = [0,0,0])
#    result_back_HR_lossless = result_back[:1000] # memory issue, only calculate 10000    
#    layerExtractor.showImg(result_back_HR_lossless,2, 'result_back_HR_lossless') 
#del forward_result
#
#
forward_result=layerExtractor.forward(dataset=trainset,datasetName='training_MNIST_0_0_1e-7'\
                                      ,mode='LR_scale_6_interpo',layer='L3',savePatch=(not testInverse),\
                                      folder = 'MNIST_weight_0_0_1e-7',weight_thresholds = [0,0,1e-7])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='LR_scale_6_interpo',layer = 'L0',folder = 'MNIST_weight_0_0_0',weight_thresholds = [0,0,0])
#    result_back_LR_lossy = result_back[:5000] # memory issue, only calculate 10000    
#    layerExtractor.showImg(result_back_LR_lossy,2, 'result_back_LR_lossy') 
del forward_result    

forward_result=layerExtractor.forward(dataset=testset,datasetName='testing_MNIST_0_0_1e-7'\
                                      ,mode='LR_scale_6_interpo',layer='L3',savePatch=(not testInverse),\
                                      folder = 'MNIST_weight_0_0_1e-7',weight_thresholds =  [0,0,1e-7])

del forward_result

#forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
#                                      ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',weight_thresholds = [0,0,0])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'MNIST_weight',weight_thresholds = [0,0,0])
#    result_back_HR_lossless = result_back[:5000] # memory issue, only calculate 10000    
#    layerExtractor.showImg(result_back_HR_lossless,2, 'result_back_HR_lossless') 
#del forward_result
#
#
#forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST_0_1e-4_0"\
#                                      ,mode='HR',layer='L3',savePatch=(not testInverse),\
#                                      folder = 'MNIST_weight_0_1e-4_0',weight_thresholds = [0,1e-4,0])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'MNIST_weight_0_0_0',weight_thresholds = [0,0,0])
#    result_back_HR_lossy = result_back[:5000] # memory issue, only calculate 10000    
#    layerExtractor.showImg(result_back_HR_lossy,2, 'result_back_HR_lossy') 
#del forward_result
#
#forward_result=layerExtractor.forward(dataset=testset,datasetName="testing_MNIST_0_1e-4_0"\
#                                      ,mode='HR',layer='L3',savePatch=(not testInverse),\
#                                      folder = 'MNIST_weight_0_1e-4_0',weight_thresholds = [0,1e-4,0])
#
#del forward_result






#def checkPCA(dst='zoom_2/HR/_L1_0.npy'):
#    import numpy as np
#    W_pca = np.load('/home/yifang/SAAK_superResolution/V4/MNIST_weight/'+dst)
#    W_pca = np.transpose(W_pca,(0,2,3,1))
#    n_comp = W_pca.shape[0]
#    n_feature = W_pca.shape[1]*W_pca.shape[2]*W_pca.shape[3]
#    W_pca = np.reshape(W_pca,(n_comp,n_feature))
#    return W_pca
#
#def testPCA(n_feature):
#    import numpy as np
#    A=np.random.rand(20,n_feature)
#    A_mean = np.mean(A,axis=1,keepdims=True)
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components=A.shape[1],svd_solver='full')
#    pca.fit(A-A_mean)
#    W=pca.components_
#    W=W*((W[:,[0]]>0)*2-1)
#    A_proj = np.dot(A-A_mean,W.T)
#    return W,A_proj,A-A_mean
#
#def PSNR(noiseImg, noiseFreeImg):
#    noiseImg=np.clip(noiseImg,0,255)
#    imdff = noiseFreeImg - noiseImg
#    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
#    
#    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
#    return np.mean(20 * np.log10(255.0 / rmse))
#
#if testInverse:  
#    result_back_HR_lossless = trainset['HR'][:5000]
#    print(PSNR(result_back_LR_lossy.data.numpy(),result_back_HR_lossless))