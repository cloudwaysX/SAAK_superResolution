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


data = np.load('./data/training_MNIST/LR_scale_4_interpo_L380.npy');
data_train = data[:20000]; data_test = data[20000:]
label = np.load('./data/training_MNIST/HR_L380.npy');
label_train = label[:20000]; label_test = label[20000:]
#data = np.load('./data/training_MNIST/LR_scale_4_interpo_L3100.npy');
#data_train = data[[7,10,12,27,44]]; data_test = data[[50]]
#label = np.load('./data/training_MNIST/HR_L3100.npy');
#label_train = label[[7,10,12,27,44]]; label_test = label[[50]]
#
data_featureNum = data_train.shape[1]
data_train_sampleNum = data_train.shape[0]*data_train.shape[2]*data_train.shape[3]
data_train_arranged =np.reshape(np.transpose(data_train,(0,2,3,1)),(data_train_sampleNum,data_featureNum))
data_test_sampleNum = data_test.shape[0]*data_test.shape[2]*data_test.shape[3]
data_test_arranged =np.reshape(np.transpose(data_test,(0,2,3,1)),(data_test_sampleNum,data_featureNum))

label_featureNum = label_train.shape[1]
label_train_sampleNum = label_train.shape[0]*label_train.shape[2]*label_train.shape[3]
label_train_arranged =np.reshape(np.transpose(label_train,(0,2,3,1)),(label_train_sampleNum,label_featureNum))
label_test_sampleNum = label_test.shape[0]*label_test.shape[2]*label_test.shape[3]
label_test_arranged =np.reshape(np.transpose(label_test,(0,2,3,1)),(label_test_sampleNum,label_featureNum))

A = np.linalg.lstsq(data_train_arranged,label_train_arranged);

pred_arranged = np.dot(data_test_arranged,A[0])
#diff = pred_arranged - label_arranged;
pred = np.reshape(pred_arranged,(data_test.shape[0],data_test.shape[2],data_test.shape[3],data_featureNum))
pred = np.transpose(pred,(0,3,1,2))

pred_var = torch.from_numpy(pred)
pred_var = Variable(pred_var)
#
label_var = torch.from_numpy(label_test)
label_var = Variable(label_var)
#
result_back_pred = layerExtractor.inverse(pred_var,mode='HR',layer = 'L3',folder = 'MNIST_weight',keepComp=[1,0.9,0.8])
layerExtractor.showImg(result_back_pred,3, 'result_back_pred')

result_back_label = layerExtractor.inverse(label_var,mode='HR',layer = 'L3',folder = 'MNIST_weight',keepComp=[1,0.9,0.8])
layerExtractor.showImg(result_back_label,3, 'result_back_label')

#label_var = torch.from_numpy(label)
#label_var = Variable(label_var)
#result_back_label = layerExtractor.inverse(label_var,mode='HR',layer = 'L3',folder = 'MNIST_weight')
#for i in range(60,80):
#    layerExtractor.showImg(result_back_label,i, 'result_back_label')
#    
#
data_var = torch.from_numpy(data_test)
data_var = Variable(data_var)
result_back_data = layerExtractor.inverse(data_var,mode='LR_scale_4_interpo',layer = 'L3',folder = 'MNIST_weight',keepComp=[1,0.9,0.8])
layerExtractor.showImg(result_back_data,0, 'result_back_data')