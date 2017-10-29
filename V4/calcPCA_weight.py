#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:02:03 2017

@author: yifang
"""

from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
import numpy as np


def Cal_W_PCA(X,weight_threshold,reception_size = 2,stride = (2,2,3)):
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
    
    X_aranged_whiten = X_aranged - np.mean(X_aranged,axis=1,keepdims=True)
    ############### AC ####################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged_whiten.shape[1],svd_solver='full')
    pca.fit(X_aranged_whiten)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
#    print(pca.explained_variance_ratio_)
    n_keptComponent = pca.explained_variance_ratio_ > weight_threshold
    W_pca_aranged = W_pca_aranged*((W_pca_aranged[:,[0]]>0)*2-1) #if the first item of W_pca_arranged is negative, make the whole vector posistive
    if weight_threshold !=0: #keep the DC weight
        W_pca_aranged = np.concatenate((W_pca_aranged[n_keptComponent],W_pca_aranged[[-1],:]))

    
    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(W_pca_aranged.shape[0],reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    
    return W_pca, W_pca_aranged.shape[0]


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.manual_seed(1)

    
class Net(nn.Module):

    def __init__(self,weight_threshold,zoomFactor):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.dc={'dc1':None,'dc2':None,'dc3':None,'dc4':None,'dc5':None}
        self.upsample = {'upsample1':None,'upsample2':None,'upsample3':None,'upsample4':None,'upsample5':None}
        self.conv = {'conv1':None,'conv2':None,'conv3':None,'conv4':None,'conv5':None}
        self.inChannel = {'in1':1,'in2':None,'in3':None,'in4':None,'in5':None}
        self.outChannel = {'out1':None,'out2':None,'out3':None,'out4':None,'out5':None}
        self.zoomFactor = zoomFactor;
    
    
    def updateLayers(self,curL_in,curL_out,weight,layer):
        print('updating layer: '+layer)
        self.dc['dc'+layer]=nn.Conv2d(curL_in, 1, self.zoomFactor,stride=self.zoomFactor ) 
        self.upsample['upsample'+layer] = nn.Upsample(scale_factor=self.zoomFactor)
        self.conv['conv'+layer] = nn.Conv2d(curL_in, curL_out, self.zoomFactor,stride=self.zoomFactor ) 
        self.conv['conv'+layer].weight.data= torch.Tensor(weight)
        self.conv['conv'+layer].bias.data.fill_(0)
        self.dc['dc'+layer].weight.data = torch.Tensor(weight[[-1],:,:,:])
        self.dc['dc'+layer].bias.data.fill_(0)
        self.inChannel['in'+layer] = curL_in;
        self.outChannel['out'+layer] = curL_out;
        
    def forward(self,input,layer):
        print('forwarding at layer: '+layer)
        z_DC = self.dc['dc'+layer](input)
        n_feature = input.size()[1]*self.zoomFactor*self.zoomFactor
        z_mean = self.upsample['upsample'+layer](z_DC/np.sqrt(n_feature))
        z_AC = self.conv['conv'+layer](input-z_mean)
        z_AC = z_AC[:,:-1,:,:]
        A_1 = F.relu(z_AC);A_2=F.relu(-z_AC)
        A = torch.cat((z_DC,A_1,A_2),dim = 1)
        return A
    
def calcW(dataset,weight_thresholds = [0,0,0],mode='HR',folder='weight',zoomfactor = 2):
    import os
    folder = folder + '/zoom_'+str(zoomfactor)
    if not os.path.exists('./'+folder+'/'+mode):
        os.makedirs('./'+folder+'/'+mode)
        
    net = Net(weight_thresholds,zoomfactor)
    
    X=torch.Tensor(dataset[mode]) #mode can be 'HR', 'LR_scale_X'
    X=Variable(X)
    curInCha = 1;
    W_pca,curOutChar= Cal_W_PCA(X.data.numpy(),weight_threshold=weight_thresholds[0],stride = (zoomfactor,zoomfactor,curInCha))
    np.save('./'+folder+'/'+mode+'/_L1_'+str(weight_thresholds[0])+'.npy',W_pca)
    net.updateLayers(curInCha,curOutChar,W_pca,'1')
    curInCha = curOutChar*2-1
    A1 = net.forward(X,'1'); del X
#    print(curOutChar); print(A1.size())
    W_pca,curOutChar= Cal_W_PCA(A1.data.numpy(),weight_threshold=weight_thresholds[1],stride = (zoomfactor,zoomfactor,curInCha))
    np.save('./'+folder+'/'+mode+'/_L2_'+str(weight_thresholds[1])+'.npy',W_pca)
    net.updateLayers(curInCha,curOutChar,W_pca,'2')
    curInCha = curOutChar*2-1
    A2 = net.forward(A1,'2');del A1
    W_pca,curOutChar= Cal_W_PCA(A2.data.numpy(),weight_threshold=weight_thresholds[2],stride = (zoomfactor,zoomfactor,curInCha))
    np.save('./'+folder+'/'+mode+'/_L3_'+str(weight_thresholds[2])+'.npy',W_pca)
    net.updateLayers(curInCha,curOutChar,W_pca,'3')
    
    np.save('./'+folder+'/'+mode+'/'+str(weight_thresholds[0])+'_'+str(weight_thresholds[1])+'_'+\
            str(weight_thresholds[2])+'_struc.npy',(net.inChannel,net.outChannel))

    
    
#    curInCha = curOutChar*8-1
#    A3 = net.forward(A2,'3');
#    del A2  
    



