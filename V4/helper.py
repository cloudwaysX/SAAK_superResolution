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
myDataset.readMNIST()
trainset = myDataset.loadData(dataset = 'training')
#testset = myDataset.loadData(dataset = 'testing')

needCalcWeight = True
testInverse = True

if needCalcWeight:
    calcPCA_weight.calcW(trainset,mode='HR',folder = 'MNIST_weight',keepComp = [1,1,1])
#    calcPCA_weight.calcW(trainset,mode='LR_scale_4_interpo',folder = 'MNIST_weight',keepComp = [1,1,0.8])
#    calcPCA_weight.calcW(trainset,mode='LR_scale_6_interpo',folder = 'MNIST_weight')
    
#forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
#                                      ,mode='LR_scale_4_interpo',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',keepComp = [1,1,0.8])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='LR_scale_4_interpo',layer = 'L3',folder = 'MNIST_weight',keepComp = [1,1,0.8])
#    result_back_LR = result_back[:1000] # memory issue, only calculate 10000
#    layerExtractor.showImg(result_back_LR,2, 'result_back_LR')
#
#del forward_result


#forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
#                                      ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',keepComp = [1,1,1])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'MNIST_weight',keepComp = [1,1,1])
#    result_back_HR_lossless = result_back[:1000] # memory issue, only calculate 10000    
#    layerExtractor.showImg(result_back_HR_lossless,2, 'result_back_HR_lossless') 
#del forward_result
#
#
#forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
#                                      ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',keepComp = [1,1,0.8])
#if testInverse:    
#    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'MNIST_weight',keepComp = [1,1,0.8])
#    result_back_HR_lossy = result_back[:1000] # memory issue, only calculate 10000    
#    layerExtractor.showImg(result_back_HR_lossy,2, 'result_back_HR_lossy') 
#del forward_result
#    
#
#
##A1=forward_result1.data.numpy()[0];A2=forward_result2.data.numpy()[0]
##B1=result_back1.data.numpy()[0];B2=result_back2.data.numpy()[0];
#def checkPCA(dst='HR/_L1_100.npy'):
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
#    imdff = noiseFreeImg - noiseImg
#    imdff = imdff.data.numpy()
#    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
#    
#    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
#    return np.mean(20 * np.log10(255.0 / rmse))
#
#print(PSNR(result_back_HR_lossy,result_back_HR_lossless))