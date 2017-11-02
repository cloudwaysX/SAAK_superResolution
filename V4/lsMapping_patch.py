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
from time import clock



import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("action", choices=['kmean','hierachy_kmean'])
parser.add_argument("--cluster_num", required=True,type=int, help="number of cluster we want to seperate")
parser.add_argument("--scale", default=6,type=int, help="super resolution downsacle factor")
#parser.add_argument("--interpo", action="store_true", help="interpolation?")
parser.add_argument("--LR_folder",default="_MNIST_0_0_1e-7",help = 'the folder storing LR feature vector')
parser.add_argument("--HR_folder",default="_MNIST_0_1e-4_0",help = 'the folder storing HR feature vector')
parser.add_argument("--LR_stop_layer",default="_L3_1e-07",help = 'at which LR layer we stopped')
parser.add_argument("--HR_stop_layer",default="_L3_0",help = 'at which HR layer we stopped')

opt = parser.parse_args()

def readData():
    data_train= np.load('./data/training'+opt.LR_folder+"/LR_scale_"+str(opt.scale)+"_interpo"+opt.LR_stop_layer+".npy")
    data_test = np.load('./data/testing'+opt.LR_folder+"/LR_scale_"+str(opt.scale)+"_interpo"+opt.LR_stop_layer+".npy")[:1000]
    label_train = np.load('./data/training'+opt.HR_folder+"/HR"+opt.HR_stop_layer+'.npy')
    #layber test is the final test HR image instead of L3 of test HR image
    label_test = np.load('./data/testing'+opt.HR_folder+"/HR_L0.npy")[:1000]   
    return {'data_train':data_train,'data_test':data_test,'label_train':label_train,'label_test':label_test}

def PSNR(noiseImg, noiseFreeImg,lowThreshold = 0):
    noiseImg=np.clip(noiseImg,0,255)
    noiseImg[noiseImg<lowThreshold]=0
    imdff = noiseFreeImg - noiseImg
    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
    
    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
    return np.mean(20 * np.log10(255.0 / rmse))

def hierachyKmean_classifier(data_train_arranged_cluster_use,train_label):
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    clf = NearestCentroid()
    clf.fit(data_train_arranged_cluster_use,train_label)
    return clf

    
#def main():
tic=clock()
# =============================================================================
print('reading data...')
data = readData()
data_featureNum = data['data_train'].shape[1]
data_train_sampleNum = data['data_train'].shape[0]*data['data_train'].shape[2]*data['data_train'].shape[3]
data_train_arranged =np.reshape(np.transpose(data['data_train'],(0,2,3,1)),(data_train_sampleNum,data_featureNum))
print('shape of training data is:',data['data_train'].shape)

data_test_sampleNum = data['data_test'].shape[0]*data['data_test'].shape[2]*data['data_test'].shape[3]
data_test_arranged =np.reshape(np.transpose(data['data_test'],(0,2,3,1)),(data_test_sampleNum,data_featureNum))
print('shape of testing data is:',data['data_test'].shape)

label_featureNum = data['label_train'].shape[1]
label_train_sampleNum = data['label_train'].shape[0]*data['label_train'].shape[2]*data['label_train'].shape[3]
label_train_arranged =np.reshape(np.transpose(data['label_train'],(0,2,3,1)),(label_train_sampleNum,label_featureNum))
print('shape of training label is:',data['label_train'].shape)

# =============================================================================    
print('we are using classification method: ',opt.action)

#if not conv to the last state

if opt.action == 'kmean':
    from sklearn.cluster import KMeans
    k_means = KMeans(n_clusters = opt.cluster_num,max_iter=300)
#    k_means.fit(data_train_arranged_cluster_use)
    k_means.fit(data_train_arranged[:40000]) #truncate size!!!!!!!!!!!
#    train_label = k_means.labels_
    train_label=k_means.predict(data_train_arranged) #truncate size!!!!!!!!!!!
else:
    from sklearn.cluster import AgglomerativeClustering
    k_means = AgglomerativeClustering(n_clusters=opt.cluster_num, affinity='euclidean',linkage='ward')
    # k_means.fit(data_train_arranged_cluster_use)
    k_means.fit(data_train_arranged[:40000]) #truncate size!!!!!!!!!!!
    train_label = k_means.labels_
    clf = hierachyKmean_classifier(data_train_arranged[:40000],train_label) #truncate size!!!!!!!!!!!
    train_label = clf.predict(data_train_arranged) #truncate size!!!!!!!!!!!
    
print('finish classifier traning')

A={}
for i in range(opt.cluster_num):
    A[str(i)]=np.linalg.lstsq(data_train_arranged[train_label==i],label_train_arranged[train_label==i])
       
data_test_arranged_cluster_use = np.reshape(data_test_arranged,[int(data_test_sampleNum/16),data_featureNum*16])
if opt.action == 'kmean':
    test_label=k_means.predict(data_test_arranged)
else:
    clf = hierachyKmean_classifier(data_train_arranged,train_label)
    test_label = clf.predict(data_test_arranged)
    

# ============================================================================= 
final_pred_arranged = np.zeros((data['label_test'].shape[0]*16,label_featureNum))
for i in range(opt.cluster_num):
    pred_arranged = np.dot(data_test_arranged[test_label==i],A[str(i)][0])
    final_pred_arranged[test_label==i] = pred_arranged
    
    
pred = np.reshape(final_pred_arranged,(int(final_pred_arranged.shape[0]/16),data['data_test'].shape[2],data['data_test'].shape[3],label_featureNum))
pred = np.transpose(pred,(0,3,1,2))    
    
pred_var = torch.Tensor(pred)
pred_var = Variable(pred_var)
#
label_var = torch.from_numpy(data['label_test'])
label_var = Variable(label_var)
#
result_back_pred = layerExtractor.inverse(pred_var,mode='HR',layer = 'L3',folder = 'MNIST_weight_0_1e-4_0',weight_thresholds = [0,1e-4,0])

total_PSNR=PSNR(result_back_pred.data.numpy(),data['label_test'],lowThreshold=0)

toc = clock()
    
print('total_PSNR',total_PSNR)
print('time',toc-tic)

def showImg(index=0):
    layerExtractor.showImg(label_var,index, 'result_back_label')
    layerExtractor.showImg(result_back_pred,index, 'result_back_pred')
    
    

