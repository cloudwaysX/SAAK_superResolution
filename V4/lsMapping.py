#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:38:33 2017

@author: yifang
"""
import numpy as np
import layerExtractor 
import torch
from torch.autograd import Variable


data_train= np.load('./data/training_MNIST_0_0_1e-7/LR_scale_6_interpo_L3_1e-07.npy')
data_test = np.load('./data/testing_MNIST_0_0_1e-7/LR_scale_6_interpo_L3_1e-07.npy')[:1000]
label_train = np.load('./data/training_MNIST_0_1e-4_0/HR_L3_0.npy')
label_test = np.load('./data/testing_MNIST_0_1e-4_0/HR_L0.npy')[:1000]

data_featureNum = data_train.shape[1]
data_train_sampleNum = data_train.shape[0]*data_train.shape[2]*data_train.shape[3]
data_train_arranged =np.reshape(np.transpose(data_train,(0,2,3,1)),(data_train_sampleNum,data_featureNum))
data_test_sampleNum = data_test.shape[0]*data_test.shape[2]*data_test.shape[3]
data_test_arranged =np.reshape(np.transpose(data_test,(0,2,3,1)),(data_test_sampleNum,data_featureNum))

label_featureNum = label_train.shape[1]
label_train_sampleNum = label_train.shape[0]*label_train.shape[2]*label_train.shape[3]
label_train_arranged =np.reshape(np.transpose(label_train,(0,2,3,1)),(label_train_sampleNum,label_featureNum))

A = np.linalg.lstsq(data_train_arranged,label_train_arranged);

pred_arranged = np.dot(data_test_arranged,A[0])
pred = np.reshape(pred_arranged,(data_test.shape[0],data_test.shape[2],data_test.shape[3],label_featureNum))
pred = np.transpose(pred,(0,3,1,2))

pred_var = torch.from_numpy(pred)
pred_var = Variable(pred_var)
#
label_var = torch.from_numpy(label_test)
label_var = Variable(label_var)
#
result_back_pred = layerExtractor.inverse(pred_var,mode='HR',layer = 'L3',folder = 'MNIST_weight_0_1e-4_0',weight_thresholds = [0,1e-4,0])
layerExtractor.showImg(result_back_pred,3, 'result_back_pred',lowThreshold = 0)


label_test_var = Variable(torch.from_numpy(label_test))
layerExtractor.showImg(label_test_var,3, 'result_back_label')

#data_var = torch.from_numpy(data_test)
#data_var = Variable(data_var)
#result_back_data = layerExtractor.inverse(data_var,mode='LR_scale_4_interpo',layer = 'L3',folder = 'MNIST_weight_0_0_0',weight_thresholds = [0,0,0])
#layerExtractor.showImg(result_back_data,9, 'result_back_data')

def PSNR(noiseImg, noiseFreeImg,lowThreshold = 0):
    noiseImg=np.clip(noiseImg,0,255)
    noiseImg[noiseImg<lowThreshold]=0
    imdff = noiseFreeImg - noiseImg
    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
    
    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
    return np.mean(20 * np.log10(255.0 / rmse))


print(PSNR(result_back_pred.data.numpy(),label_test,lowThreshold=50))
#    


#import numpy as np
#HR = np.load('./data/training_MNIST/HR_L0.npy')[50000:]
#LR = np.load('./data/training_MNIST/LR_scale_6_interpo_L0.npy')[50000:]
#def PSNR(noiseImg, noiseFreeImg):
#    imdff = noiseFreeImg - noiseImg
#    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
#    
#    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
#    return np.mean(20 * np.log10(255.0 / rmse))
#
#print(PSNR(LR,HR))