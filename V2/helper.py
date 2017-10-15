#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:07:36 2017

@author: yifang
"""

import readMNIST
import LayerExtractor

myDataset = readMNIST.DatasetMNIST()
myDataset.readMNIST()
trainset = myDataset.loadData(dataset = 'training')
#testset = myDataset.loadData(dataset = 'testing')

needCalcWeight = False

if needCalcWeight:
    import calcPCA_weight
    calcPCA_weight.calcW(trainset,mode='HR')
    calcPCA_weight.calcW(trainset,mode='LR')
    
LayerExtractor.forward(dataset=trainset,datasetName="training",mode='LR',layer='L3')
#LayerExtractor.inverse(forward_result=A3,net=net)

#
LayerExtractor.forward(dataset=trainset,datasetName="training",mode='HR',layer='L3')
#LayerExtractor.inverse(forward_result=A3,net=net)