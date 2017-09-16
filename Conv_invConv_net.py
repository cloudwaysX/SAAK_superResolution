#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 01:56:38 2017

@author: yifang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:13:01 2017

@author: yifang
"""
import argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--weight_INT', type=str, default='pca', metavar='N',
                    help='choose what kind of weight initialization method you want to use')


args = parser.parse_args()

import numpy as np
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from sklearn import preprocessing

def Cal_inv_W(W,reception_size = 2,stride = (2,2,3)):
    # W has shape (outputChannel, inputChannel, receptionSize, receptionSize)
    assert len(W.shape)==4, "input weight is not 4D"
    assert W.shape[2] == W.shape[3], "reception window is not square"
    reception_size = W.shape[2]    
    n_input_channels = W.shape[1]
    n_output_channels = W.shape[0]
    
    # rearranged the weight
    # shape is (n_components, n_features)
    featureNum = reception_size**2*n_input_channels
    W_aranged = np.transpose(W,(0,2,3,1))
    W_aranged = np.reshape(W_aranged,(int(featureNum),int(featureNum)))
 
    
    #######################################
    #calculate the inverse weight
    W_aranged_INV =  np.linalg.inv(W_aranged).T
    W_INV = np.reshape(W_aranged_INV,(n_output_channels,reception_size,reception_size,n_input_channels))
    W_INV = np.transpose(W_INV,(0,3,1,2))

  
    return W_INV

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


#X = np.arange(3*8*8,dtype=float).reshape(3,8,8)
#X = np.stack((X,X))
X = np.random.randn(1,3,8,8)
print(X)


#X=np.random.rand(3,8,8)*10
#W_pca, W_pca_INV, X_proj_reference, X_proj_INV_reference= Cal_W_PCA(X)
#W_pca1,W_pca_INV1= Cal_W_PCA(X)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 12, 2,stride=2 ) 
        self.conv2 = nn.Conv2d(24, 96, 2,stride=2)
        self.conv3 = nn.Conv2d(192, 768, 2,stride=2)
        self.conv4 = nn.Conv2d(192*8, 768*8, 2,stride=2)
        self.conv5 = nn.Conv2d(192*64, 768*64, 2,stride=2)

    def forward1(self, x):
        z1 = self.conv1(x)
        z1 = torch.cat((z1,-z1),dim=1)
        A1 = F.relu(z1)
        return A1
    
    def forward2(self,A1):
        z2 = self.conv2(A1)
        z2 = torch.cat((z2,-z2),dim=1)
        A2 = F.relu(z2)
        return A2
        
    def forward3(self,A2):
        z3 = self.conv3(A2)
        z3 = torch.cat((z3,-z3),dim=1)
        A3 = F.relu(z3)
        return A3
    
    def forward4(self,A3):
        z4 = self.conv4(A3)
        z4 = torch.cat((z4,-z4),dim=1)
        A4 = F.relu(z4)
        return A4
    
    def forward5(self,A4):
        z5 = self.conv5(A4)
        z5 = torch.cat((z5,-z5),dim=1)
        A5 = F.relu(z5)
        return A5

net = Net()
print('processing A1 ...')
#W_pca, W_pca_INV, X_proj_reference, X_proj_INV_reference= Cal_W_PCA(X)W_pca1= np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_12_3_2_2.npy',W_pca1)
# wrap input as variable
X=torch.Tensor(X)
X=Variable(X)
if args.weight_INT == 'pca':
    W_pca1=np.load('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_12_3_2_2.npy')
    net.conv1.weight.data=torch.Tensor(W_pca1)
net.conv1.bias.data.fill_(0)
A1 = net.forward1(X)

print('processing A2 ...')
if args.weight_INT == 'pca':
    W_pca2=np.load('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_96_24_2_2.npy')
    net.conv2.weight.data=torch.Tensor(W_pca2)
net.conv2.bias.data.fill_(0)
A2 = net.forward2(A1)

print('processing A3 ...')
if args.weight_INT == 'pca':
    W_pca3=np.load('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_768_192_2_2.npy')
    net.conv3.weight.data=torch.Tensor(W_pca3)
net.conv3.bias.data.fill_(0)
A3 = net.forward3(A2)

#print('processing A4 ...')
#W_pca4= Cal_W_PCA(A3.data.numpy(),stride = (2,2,192*8))
#net.conv4.weight.data=torch.Tensor(W_pca4)
#net.conv4.bias.data.fill_(0)
#A4 = net.forward4(A3)

class Inv_Net(nn.Module):

    def __init__(self):
        super(Inv_Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.deconv1 = nn.ConvTranspose2d(12,3,2,stride=2 )
        self.deconv2 = nn.ConvTranspose2d(96, 24, 2,stride=2)
        self.deconv3 = nn.ConvTranspose2d(768, 192, 2,stride=2)
        self.deconv4 = nn.ConvTranspose2d(768*8, 768, 2,stride=2)
        self.deconv5 = nn.ConvTranspose2d(768*64, 768*8, 2,stride=2)

    def forward1_inv(self, A1):
        z1 = A1[:,0:12,:,:]-A1[:,12:24,:,:] 
        x = self.deconv1(z1)
        return x
        
    def forward2_inv(self, A2):
        z2 = A2[:,0:96,:,:]-A2[:,96:192,:,:] 
        A1 = self.deconv2(z2)
        return A1
    
    def forward3_inv(self,A3):
        z3 = A3[:,0:768,:,:]-A3[:,768:768*2,:,:] 
        A2 = self.deconv3(z3)
        return A2
    
    def forward4_inv(self,A4):
        z4 = A4[:,0:768*8,:,:]-A4[:,768*8:768*2*8,:,:] 
        A3 = self.deconv4(z4)
        return A3    
    
    def forward5_inv(self,A5):
        z5 = A5[:,0:768*64,:,:]-A5[:,768*64:768*2*64,:,:] 
        A4 = self.deconv5(z5)
        return A4    
    
net_inv = Inv_Net()

import time
tic = time.clock()


#A4_back = A4
#net_inv.deconv4.weight.data=torch.Tensor(W_pca4)
#net_inv.deconv4.bias.data.fill_(0)
#A3_back = net_inv.forward4_inv(A4_back)

A3_back=A3
print('processing A3 back to A2 ...')
if args.weight_INT == 'pca':
    net_inv.deconv3.weight.data=net.conv3.weight.data
else:
    net_inv.deconv3.weight.data=torch.Tensor(Cal_inv_W(net.conv3.weight.data.numpy()))
net_inv.deconv3.bias.data.fill_(0)
A2_back = net_inv.forward3_inv(A3_back)

print('processing A2 back to A1...')
if args.weight_INT == 'pca':
    print('pca')
    net_inv.deconv2.weight.data=net.conv2.weight.data
else:
    net_inv.deconv2.weight.data=torch.Tensor(Cal_inv_W(net.conv2.weight.data.numpy()))
net_inv.deconv2.bias.data.fill_(0)
A1_back = net_inv.forward2_inv(A2_back)

#A1_back = A1
print('processing A1 back to x ...')
if args.weight_INT == 'pca':
    net_inv.deconv1.weight.data=net.conv1.weight.data
else:
    net_inv.deconv1.weight.data=torch.Tensor(Cal_inv_W(net.conv1.weight.data.numpy()))
net_inv.deconv1.bias.data.fill_(0)
result_back = net_inv.forward1_inv(A1_back)
print(result_back.data[0,:,:,:])
#

toc = time.clock()
print("Total time is %f" %(toc-tic))