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

def Cal_W_PCA_full(X,reception_size = 2,stride = (2,2,3)):
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
    outChannel = W_pca_aranged.shape[1]


    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(outChannel,reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    
    return W_pca

def Cal_W_PCA_reduceD(X_AC,n_keptComponent,reception_size = 1,stride = (1,1,3)):
#    print(n_keptComponent)
    assert len(X_AC.shape)==4, "input images batch is not 4D"
      
    X_AC=np.transpose(X_AC,(2,3,0,1)) # width,height,sampleNum,depth
    assert X_AC.shape[0] == X_AC.shape[1], "input image is not square"
    input_width = X_AC.shape[0]    
    input_depth = X_AC.shape[3]
    sampleNum = X_AC.shape[2]
    
    #######################################
    # extracted pathes from X, the shape of X_arranged is 
    #(input_width/stride,input_width/stride,1,reception_size,reception_size,input_depth)
    X_AC = np.reshape(X_AC,(input_width,input_width,input_depth*sampleNum)) #stack all samples alonge depth axis
    X_aranged_AC = view_as_windows(X_AC, window_shape=(reception_size, reception_size,input_depth),step=stride)

    # rearranged the extracted patches stacks
    # shape is (n_samples, n_features)
    patchNum = ((input_width-reception_size)/stride[0]+1)**2*sampleNum
    featureNum = reception_size**2*input_depth
    X_aranged_AC = np.reshape(X_aranged_AC, (int(patchNum), int(featureNum)))
    
    ############### AC ####################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged_AC.shape[1],svd_solver='full')
    pca.fit(X_aranged_AC)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
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

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        keepComp_init = 1
        curL_in = 1; curL_out = round(curL_in*4*keepComp_init) # initial
        self.avgPool1 = nn.AvgPool3d ((curL_in,2,2),stride=(curL_in,2,2))
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(curL_in, curL_in*4, 2,stride=2 ) 
        self.conv1_reduceD = nn.Conv2d(curL_in*4, curL_out, 1,stride=1 )
        self.downsample1 = nn.MaxPool2d(2,stride=2)
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=1)
        
        self.avgPool2 = nn.AvgPool3d ((curL_in,2,2),stride=(curL_in,2,2))
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(curL_in, curL_in*4, 2,stride=2 ) 
        self.conv2_reduceD = nn.Conv2d(curL_in*4, curL_out, 1,stride=1 )
        self.downsample2 = nn.MaxPool2d(2,stride=2)
        
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=1)
        self.avgPool3 = nn.AvgPool3d ((curL_in,2,2),stride=(curL_in,2,2))
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(curL_in, curL_in*4, 2,stride=2 ) 
        self.conv3_reduceD = nn.Conv2d(curL_in*4, curL_out, 1,stride=1 )
        self.downsample3 = nn.MaxPool2d(2,stride=2)


    def forward1(self, x):
        z1_DC = self.avgPool1(torch.unsqueeze(x,dim=1))
        z1_DC = torch.squeeze(z1_DC,dim=1)
        z1_DC = self.upsample1(z1_DC)
        z1_AC = self.conv1(x-z1_DC)
        return {'AC':z1_AC, 'DC':z1_DC}
    
    def foward1_reduceD(self,z1_AC,z1_DC):
        z1_AC = self.conv1_reduceD(z1_AC)
        z1_DC = self.downsample1(z1_DC)
        A1_1 = F.relu(z1_AC);A1_2=F.relu(-z1_AC)
        A1 = torch.cat((z1_DC,A1_1,A1_2),dim = 1)
        return A1
    
    def forward2(self, A1):
        z2_DC = self.avgPool2(torch.unsqueeze(A1,dim=1))
        z2_DC = torch.squeeze(z2_DC,dim=1)
        z2_DC = self.upsample2(z2_DC)
        z2_AC = self.conv2(A1-z2_DC)
        return {'AC':z2_AC, 'DC':z2_DC}
    
    def foward2_reduceD(self,z2_AC,z2_DC):
        z2_AC = self.conv2_reduceD(z2_AC)
        z2_DC = self.downsample2(z2_DC)
        A2_1 = F.relu(z2_AC);A2_2=F.relu(-z2_AC)
        A2 = torch.cat((z2_DC,A2_1,A2_2),dim = 1)
        return A2
    
    def forward3(self, A2):
        z3_DC = self.avgPool3(torch.unsqueeze(A2,dim=1))
        z3_DC = torch.squeeze(z3_DC,dim=1)
        z3_DC = self.upsample3(z3_DC)
        z3_AC = self.conv3(A2-z3_DC)
        return {'AC':z3_AC, 'DC':z3_DC}
    
    def foward3_reduceD(self,z3_AC,z3_DC):
        z3_AC = self.conv3_reduceD(z3_AC)
        z3_DC = self.downsample3(z3_DC)
        A3_1 = F.relu(z3_AC);A3_2=F.relu(-z3_AC)
        A3 = torch.cat((z3_DC,A3_1,A3_2),dim = 1)
        return A3
    
def calcW (dataset,mode='HR'):
    net = Net()
    # wrap input as variable
    X=torch.Tensor(dataset[mode])
    X=Variable(X)
    print('processing A1 ...')
    curInCha,curOutCha = net.conv1.weight.size()[1],net.conv1.weight.size()[0]
    W_pca1= Cal_W_PCA_full(X.data.numpy(),stride = (2,2,curInCha))
    #store the weight
    np.save('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L1full.npy',W_pca1)
    net.conv1.weight.data=torch.Tensor(W_pca1)
    net.conv1.bias.data.fill_(0)
    z1 = net.forward1(X); del X
    curInCha,curOutCha = net.conv1_reduceD.weight.size()[1],net.conv1_reduceD.weight.size()[0]
    W_pca1_reduceD = Cal_W_PCA_reduceD(z1['AC'].data.numpy(),curOutCha,stride = (1,1,curInCha))
    np.save('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L1reduceD.npy',W_pca1_reduceD)
    net.conv1_reduceD.weight.data = torch.Tensor(W_pca1_reduceD)
    net.conv1_reduceD.bias.data.fill_(0)
    A1 = net.foward1_reduceD(z1['AC'],z1['DC']); del z1
    #
    print('processing A2 ...')
    curInCha,curOutCha = net.conv2.weight.size()[1],net.conv2.weight.size()[0]
    W_pca2= Cal_W_PCA_full(A1.data.numpy(),stride = (2,2,curInCha))
    #store the weight
    np.save('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L2full.npy',W_pca2)
    net.conv2.weight.data=torch.Tensor(W_pca2)
    net.conv2.bias.data.fill_(0)
    z2 = net.forward2(A1); del A1
    curInCha,curOutCha = net.conv2_reduceD.weight.size()[1],net.conv2_reduceD.weight.size()[0]
    W_pca2_reduceD = Cal_W_PCA_reduceD(z2['AC'].data.numpy(),curOutCha,stride = (1,1,curInCha))
    np.save('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L2reduceD.npy',W_pca2_reduceD)
    net.conv2_reduceD.weight.data = torch.Tensor(W_pca2_reduceD)
    net.conv2_reduceD.bias.data.fill_(0)
    A2 = net.foward2_reduceD(z2['AC'],z2['DC']); del z2
    #
    print('processing A3 ...')
    curInCha,curOutCha = net.conv3.weight.size()[1],net.conv3.weight.size()[0]
    W_pca3= Cal_W_PCA_full(A2.data.numpy(),stride = (2,2,curInCha))
    #store the weight
    np.save('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L3full.npy',W_pca3)
    #reduce mean
    W_pca3= W_pca3 - 1/(2*2*curInCha)
    net.conv3.weight.data=torch.Tensor(W_pca3)
    net.conv3.bias.data.fill_(0)
    z3 = net.forward3(A2); del A2
    curInCha,curOutCha = net.conv3_reduceD.weight.size()[1],net.conv3_reduceD.weight.size()[0]
    W_pca3_reduceD = Cal_W_PCA_reduceD(z3['AC'].data.numpy(),curOutCha,stride = (1,1,curInCha))
    np.save('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L3reduceD.npy',W_pca3_reduceD)
    net.conv3_reduceD.weight.data = torch.Tensor(W_pca3_reduceD)
    net.conv3_reduceD.bias.data.fill_(0)
#    A3 = net.foward3_reduceD(z3['AC'],z3['DC']); del z3
#    torch.save(net.state_dict(),'/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST.pth')