#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:07:36 2017

@author: yifang
"""

import readDataset
import layerExtractor
import calcPCA_weight


def PSNR(noiseImg, noiseFreeImg):
    import numpy as np
    noiseImg=np.clip(noiseImg,0,255)
    imdff = noiseFreeImg - noiseImg
    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
    
    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
    return np.mean(20 * np.log10(255.0 / rmse))

 
needCalcWeight = True
testInverse = False

myDataset = readDataset.DatasetBSD()
myDataset.readBSD()
trainset = myDataset.loadData(dataset = 'training')

if needCalcWeight:
    calcPCA_weight.calcW(trainset,mode='HR',folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5],printPCAratio = False)
    calcPCA_weight.calcW(trainset,mode='LR_scale_4_interpo',folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5],printPCAratio = False)
    
forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_BSD_0_0_1e-5",printNet=True\
                                       ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5],)
if testInverse:    
    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5]);del forward_result
    result_back_HR_lossy = result_back[:1000] # memory issue, only calculate 1000    
#    layerExtractor.showImg(result_back_HR_lossy,4, 'result_back_HR_lossless') 
else:
    del forward_result
                                      
 
 
forward_result=layerExtractor.forward(dataset=trainset,datasetName='training_BSD_0_0_1e-5',printNet=True\
                                       ,mode='LR_scale_4_interpo',layer='L3',savePatch=(not testInverse),\
                                       folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5])
if testInverse:    
    result_back = layerExtractor.inverse(forward_result,mode='LR_scale_4_interpo',layer = 'L3',folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5]);del forward_result
    result_back_LR_lossy = result_back[:1000] # memory issue, only calculate 1000   
#    layerExtractor.showImg(result_back_LR_lossy,4, 'result_back_LR_lossless') 
else:
    del forward_result    

if testInverse:  
    result_back_HR_lossless = trainset['HR'][:1000]
    print(PSNR(result_back_HR_lossy.data.numpy(),result_back_HR_lossless))
    print(PSNR(result_back_LR_lossy.data.numpy(),result_back_HR_lossless))

del trainset 

myDataset.readBSD(dataset='testing')
testset = myDataset.loadData(dataset = 'testing')

forward_result=layerExtractor.forward(dataset=testset,datasetName="testing_BSD_0_0_1e-5"\
                                       ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5])
del forward_result
 
 
forward_result=layerExtractor.forward(dataset=testset,datasetName='testing_BSD_0_0_1e-5'\
                                       ,mode='LR_scale_4_interpo',layer='L3',savePatch=(not testInverse),\
                                       folder = 'BSD_weight_0_0_1e-5',weight_thresholds = [0,0,1e-5]);del forward_result

del testset 
 

 























