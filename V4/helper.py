#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:07:36 2017

@author: yifang
"""

import readMNIST
import layerExtractor
import calcPCA_weight

myDataset = readMNIST.DatasetMNIST()
myDataset.readMNIST()
trainset = myDataset.loadData(dataset = 'training')
#testset = myDataset.loadData(dataset = 'testing')

needCalcWeight = True
testInverse = False

if needCalcWeight:
    calcPCA_weight.calcW(trainset,mode='HR',folder = 'MNIST_weight',keepComp = [1,0.9,0.8])
    calcPCA_weight.calcW(trainset,mode='LR_scale_4_interpo',folder = 'MNIST_weight',keepComp = [1,0.9,0.8])
#    calcPCA_weight.calcW(trainset,mode='LR_scale_6_interpo',folder = 'MNIST_weight')
    
forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
                                      ,mode='LR_scale_4_interpo',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',keepComp = [1,0.9,0.8])
if testInverse:    
    result_back = layerExtractor.inverse(forward_result,mode='LR_scale_4_interpo',layer = 'L3',folder = 'MNIST_weight',keepComp = [1,0.9,0.8])
    layerExtractor.showImg(result_back,2, 'result_back')
del forward_result

forward_result=layerExtractor.forward(dataset=trainset,datasetName="training_MNIST"\
                                      ,mode='HR',layer='L3',savePatch=(not testInverse),folder = 'MNIST_weight',keepComp = [1,0.9,0.8])
if testInverse:    
    result_back = layerExtractor.inverse(forward_result,mode='HR',layer = 'L3',folder = 'MNIST_weight',keepComp = [1,0.9,0.8])
    layerExtractor.showImg(result_back,2, 'result_back') 




