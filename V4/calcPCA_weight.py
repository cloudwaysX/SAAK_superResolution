#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:02:03 2017

@author: yifang
"""

from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
import numpy as np

def CalcNextChannelNum(preOutNum, keepComp=1):
    nextL_in = preOutNum*2 + 1
    nextL_out = round(nextL_in*4*keepComp)
    return nextL_in,nextL_out

def Cal_W_PCA(X,n_keptComponent,reception_size = 2,stride = (2,2,3)):
#    print(n_keptComponent)
    assert len(X.shape)==4, "input images batch is not 4D"
    X=np.transpose(X,(2,3,0,1)) # width,height,sampleNum,depth
    assert X.shape[0] == X.shape[1], "input image is not square"
    input_width = X.shape[0]    
    input_depth = X.shape[3]
    sampleNum = X.shape[2]
    
    #######################################
    # extracted pathes from X, the shape of X_arranged is 
    #(input_width/stride,input_width/stride,1,reception_size,reception_size,input_depth)
    X = np.reshape(X,(input_width,input_width,input_depth*sampleNum)) #stack all samples alonge depth axis
    X_aranged = view_as_windows(X, window_shape=(reception_size, reception_size,input_depth),step=stride)

    # rearranged the extracted patches stacks
    # shape is (n_samples, n_features)
    patchNum = int(((input_width-reception_size)/stride[0]+1))**2*sampleNum
    featureNum = reception_size**2*input_depth
    X_aranged = np.reshape(X_aranged, (int(patchNum), int(featureNum)))
    
    X_aranged_whiten = X_aranged
    ############### AC ####################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged_whiten.shape[1],svd_solver='full')
    pca.fit(X_aranged_whiten)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
    W_pca_aranged = W_pca_aranged*((W_pca_aranged[:,[0]]>0)*2-1) #if the first item of W_pca_arranged is negative, make the whole vector posistive
    W_pca_aranged = W_pca_aranged[:n_keptComponent,:]

    
    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(n_keptComponent,reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    
    return W_pca


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.manual_seed(1)

class Net(nn.Module):

    def __init__(self,keepComp):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        keepComp_init = keepComp[0]
        curL_in = 1; curL_out = round(curL_in*4*keepComp_init) # initial
        self.avgPool1 = nn.AvgPool3d ((curL_in,2,2),stride=(curL_in,2,2))
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(curL_in, curL_out, 2,stride=2 ) 
        self.downsample1 = nn.MaxPool2d(2,stride=2)
        
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=keepComp[1])       
        self.avgPool2 = nn.AvgPool3d ((curL_in,2,2),stride=(curL_in,2,2))
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(curL_in, curL_out, 2,stride=2 ) 
        self.downsample2 = nn.MaxPool2d(2,stride=2)
        
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=keepComp[2])
        self.avgPool3 = nn.AvgPool3d ((curL_in,2,2),stride=(curL_in,2,2))
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(curL_in, curL_out, 2,stride=2 ) 
        self.downsample3 = nn.MaxPool2d(2,stride=2)


    def forward1(self, x):
        z1_DC = self.avgPool1(torch.unsqueeze(x,dim=1))
        z1_DC = torch.squeeze(z1_DC,dim=1)
        z1_DC = self.upsample1(z1_DC)
        z1_AC = self.conv1(x-z1_DC)
        z1_DC = self.downsample1(z1_DC)
        A1_1 = F.relu(z1_AC);A1_2=F.relu(-z1_AC)
        A1 = torch.cat((z1_DC,A1_1,A1_2),dim = 1)
        return A1
    
    def forward2(self, A1):
        z2_DC = self.avgPool2(torch.unsqueeze(A1,dim=1))
        z2_DC = torch.squeeze(z2_DC,dim=1)
        z2_DC = self.upsample2(z2_DC)
        z2_AC = self.conv2(A1-z2_DC)
        z2_DC = self.downsample2(z2_DC)
        A2_1 = F.relu(z2_AC);A2_2=F.relu(-z2_AC)
        A2 = torch.cat((z2_DC,A2_1,A2_2),dim = 1)
        return A2
    
    def forward3(self, A2):
        z3_DC = self.avgPool3(torch.unsqueeze(A2,dim=1))
        z3_DC = torch.squeeze(z3_DC,dim=1)
        z3_DC = self.upsample3(z3_DC)
        z3_AC = self.conv3(A2-z3_DC)
        z3_DC = self.downsample3(z3_DC)
        A3_1 = F.relu(z3_AC);A3_2=F.relu(-z3_AC)
        A3 = torch.cat((z3_DC,A3_1,A3_2),dim = 1)
        return A3
    
def calcW(dataset,keepComp = [1,1,1],mode='HR',folder='weight'):   
    import os
    if not os.path.exists('./'+folder+'/'+mode):
        os.makedirs('./'+folder+'/'+mode)
        
    net = Net(keepComp=keepComp)

    # wrap input as variable 
    X=torch.Tensor(dataset[mode]) #mode can be 'HR', 'LR_scale_X'
    X=Variable(X)
    print('processing A1 ...')
    curInCha,curOutCha = net.conv1.weight.size()[1],net.conv1.weight.size()[0]
    W_pca1= Cal_W_PCA(X.data.numpy(),n_keptComponent=curOutCha,stride = (2,2,curInCha))
    #store the weight
    np.save('./'+folder+'/'+mode+'/_L1_'+str(int(keepComp[0]*100))+'.npy',W_pca1)
    net.conv1.weight.data=torch.Tensor(W_pca1)
    net.conv1.bias.data.fill_(0)
    A1 = net.forward1(X); del X
    #
    print('processing A2 ...')
    curInCha,curOutCha = net.conv2.weight.size()[1],net.conv2.weight.size()[0]
    W_pca2= Cal_W_PCA(A1.data.numpy(),n_keptComponent=curOutCha,stride = (2,2,curInCha))
    #store the weight
    np.save('./'+folder+'/'+mode+'/_L2_'+str(int(keepComp[1]*100))+'.npy',W_pca2)
    net.conv2.weight.data=torch.Tensor(W_pca2)
    net.conv2.bias.data.fill_(0)
    A2 = net.forward2(A1); del A1

    #
    print('processing A3 ...')
    curInCha,curOutCha = net.conv3.weight.size()[1],net.conv3.weight.size()[0]
    W_pca3= Cal_W_PCA(A2.data.numpy(),n_keptComponent=curOutCha,stride = (2,2,curInCha))
    #store the weight
    np.save('./'+folder+'/'+mode+'/_L3_'+str(int(keepComp[2]*100))+'.npy',W_pca3)
    del A2

def showW(keepComp=[1,1,1],layer='L1',mode='HR',folder='weight'):
	if layer == 'L1':
		keepComp_singleLayer=keepComp[0]
	elif layer == 'L2':
		keepComp_singleLayer=keepComp[1]
	elif layer == 'L3':
		keepComp_singleLayer=keepComp[2]

	myPCA = np.load('./'+folder+'/'+mode+'/_'+layer+'_'+str(int(keepComp_singleLayer*100))+'.npy')

	myPCA = np.transpose(myPCA,(0,2,3,1))
	myPCA = np.reshape(myPCA,(myPCA.shape[0],myPCA.shape[1]*myPCA.shape[2]*myPCA.shape[3]))

	return myPCA


