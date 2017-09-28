#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 23:13:29 2017

@author: yifang
"""

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



#X = np.arange(3*8*8,dtype=float).reshape(3,8,8)
#X = np.stack((X,X))
import os
from scipy import misc
import matplotlib.pyplot as plt


def readImg(path,name):
    image= misc.imread(os.path.join(path,name), flatten= 0)
    return image

def showImg(outBatch,index=0,title="inverse Rsult"):
    # input: outBatch with size (imageNum, imageDepth, ImageW, ImageH)
    # index = the index of the image you want to show; tiile = string of plot title
    displayImg = outBatch[index,:,:,:].data.numpy()
    displayImg=np.uint8(displayImg)
    plt.figure()    
    plt.title(title)
    plt.imshow(np.transpose(displayImg, (1,2,0)))
    plt.show()
    plt.figure()   

def showFeatureVec(outFeatureVec,index=0,title = "Feature Vec"):
    from pylab import stem
    x_axis = range(0,A3.size()[1])
    rowNum = A3.size()[2]; colNum=A3.size()[3]
    plt.figure()    
    plt.title(title)
    for row in range(0,rowNum):
        for col in range(0,colNum):
            subplot_I= row*colNum+col+1
            plt.subplot(rowNum,colNum,subplot_I)
            stem(x_axis, outFeatureVec[0,:,row,col].data.numpy())
    
    
import numpy as np
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from sklearn import preprocessing

def CalcNextChannelNum(preOutNum, keepComp=1):
    nextL_in = preOutNum*2 + 1
    nextL_out = round(nextL_in*4*keepComp)
    return nextL_in,nextL_out
    

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


 


path = '/home/yifang/SAAK_superResolution/SRCNN/Set5'
name = 'baby_GT.bmp'

img = readImg(path,name)

img_resized = misc.imresize( img, (32,32,3),interp = 'cubic')
X = np.stack((img_resized,img_resized))
X = np.transpose(X, (0,3,1,2))
X = np.float32(X)

#X=np.random.rand(3,8,8)*10
#W_pca, W_pca_INV, X_proj_reference, X_proj_INV_reference= Cal_W_PCA(X)
#W_pca1,W_pca_INV1= Cal_W_PCA(X)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        keepComp_init = 0.6
        curL_in = 3; curL_out = round(curL_in*4*keepComp_init) # initial
        self.conv1 = nn.Conv2d(curL_in, curL_in*4+1, 2,stride=2 ) 
        self.conv1_reduceD = nn.Conv2d(curL_in*4, curL_out, 1,stride=1 )
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=0.6)
        self.conv2 = nn.Conv2d(curL_in, curL_in*4+1, 2,stride=2 )
        self.conv2_reduceD = nn.Conv2d(curL_in*4, curL_out, 1,stride=1 )
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=0.6)
        self.conv3 = nn.Conv2d(curL_in, curL_in*4+1, 2,stride=2 )
        self.conv3_reduceD = nn.Conv2d(curL_in*4, curL_out, 1,stride=1 )

    def forward1(self, x):
        z1 = self.conv1(x)
        z1_AC = z1[:,1:,:,:]; z1_DC= torch.unsqueeze(z1[:,0,:,:],dim=1)
        z1_AC = self.conv1_reduceD(z1_AC)
        z1 = torch.cat((z1_DC,z1_AC),dim = 1)
        z1 = torch.cat((z1,-z1[:,1:,:,:]),dim=1)
        A1 = F.relu(z1)
        return A1
    
    def forward2(self, A1):
        z2 = self.conv2(A1)
        z2_AC = z2[:,1:,:,:]; z2_DC= torch.unsqueeze(z2[:,0,:,:],dim=1)
        z2_AC = self.conv2_reduceD(z2_AC)
        z2 = torch.cat((z2_DC,z2_AC),dim = 1)
        z2 = torch.cat((z2,-z2[:,1:,:,:]),dim=1)
        A2 = F.relu(z2)
        return A2
    
    def forward3(self, A2):
        z3 = self.conv3(A2)
        z3_AC = z3[:,1:,:,:]; z3_DC= torch.unsqueeze(z3[:,0,:,:],dim=1)
        z3_AC = self.conv3_reduceD(z3_AC)
        z3 = torch.cat((z3_DC,z3_AC),dim = 1)
        z3 = torch.cat((z3,-z3[:,1:,:,:]),dim=1)
        A3 = F.relu(z3)
        return A3
    


net = Net()
# wrap input as variable
X=torch.Tensor(X)
X=Variable(X)
print('processing A1 ...')
W_pca1 = np.load('/home/yifang/SAAK_superResolution/PCA_weight2/l1_full_06.npy')
net.conv1.weight.data=torch.Tensor(W_pca1)
net.conv1.bias.data.fill_(0)
W_pca1_reduceD = np.load('/home/yifang/SAAK_superResolution/PCA_weight2/l1_reduceD_06.npy')
net.conv1_reduceD.weight.data = torch.Tensor(W_pca1_reduceD)
net.conv1_reduceD.bias.data.fill_(0)
A1 = net.forward1(X)

print('processing A2 ...')
W_pca2 = np.load('/home/yifang/SAAK_superResolution/PCA_weight2/l2_full_06.npy')
net.conv2.weight.data=torch.Tensor(W_pca2)
net.conv2.bias.data.fill_(0)
W_pca2_reduceD = np.load('/home/yifang/SAAK_superResolution/PCA_weight2/l2_reduceD_06.npy')
net.conv2_reduceD.weight.data = torch.Tensor(W_pca2_reduceD)
net.conv2_reduceD.bias.data.fill_(0)
A2 = net.forward2(A1)

print('processing A3 ...')
W_pca3 = np.load('/home/yifang/SAAK_superResolution/PCA_weight2/l3_full_06.npy')
net.conv3.weight.data=torch.Tensor(W_pca3)
net.conv3.bias.data.fill_(0)
W_pca3_reduceD = np.load('/home/yifang/SAAK_superResolution/PCA_weight2/l3_reduceD_06.npy')
net.conv3_reduceD.weight.data = torch.Tensor(W_pca3_reduceD)
net.conv3_reduceD.bias.data.fill_(0)
A3 = net.forward3(A2)



#
#showFeatureVec(A3)


class Inv_Net(nn.Module):

    def __init__(self):
        super(Inv_Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        curL_in,curL_out = net.conv1_reduceD.weight.size()[0],net.conv1_reduceD.weight.size()[1];
        self.deconv1_reduceD = nn.ConvTranspose2d(curL_in,curL_out,1,stride=1 )
        curL_in,curL_out = net.conv1.weight.size()[0],net.conv1.weight.size()[1];
        self.deconv1 = nn.ConvTranspose2d(curL_in-1,curL_out,2,stride=2 )
        
        curL_in,curL_out = net.conv2_reduceD.weight.size()[0],net.conv2_reduceD.weight.size()[1];
        self.deconv2_reduceD = nn.ConvTranspose2d(curL_in,curL_out,1,stride=1 )
        curL_in,curL_out = net.conv2.weight.size()[0],net.conv2.weight.size()[1];
        self.deconv2 = nn.ConvTranspose2d(curL_in-1,curL_out,2,stride=2 )
        
        curL_in,curL_out = net.conv3_reduceD.weight.size()[0],net.conv3_reduceD.weight.size()[1];
        self.deconv3_reduceD = nn.ConvTranspose2d(curL_in,curL_out,1,stride=1 )
        curL_in,curL_out = net.conv3.weight.size()[0],net.conv3.weight.size()[1];
        self.deconv3 = nn.ConvTranspose2d(curL_in-1,curL_out,2,stride=2 )


    def forward1_inv(self, A1):
        half=(A1.size()[1]-1)/2;half = int(half)
        z1 = A1[:,1:half+1,:,:]-A1[:,half+1:,:,:] 
        z1 = self.deconv1_reduceD(z1)
        x = self.deconv1(z1)
        return x
    
    def forward2_inv(self, A2):
        half=(A2.size()[1]-1)/2;half = int(half)
        z2 = A2[:,1:half+1,:,:]-A2[:,half+1:,:,:] 
        z2 = self.deconv2_reduceD(z2)
        A1 = self.deconv2(z2)
        return A1
        
    def forward3_inv(self, A3):
        half=(A3.size()[1]-1)/2;half = int(half)
        z3 = A3[:,1:half+1,:,:]-A3[:,half+1:,:,:] 
        z3 = self.deconv3_reduceD(z3)
        A2 = self.deconv3(z3)
        return A2  
    
net_inv = Inv_Net()

import time
tic = time.clock()


##A4_back = A4
##net_inv.deconv4.weight.data=torch.Tensor(W_pca4)
##net_inv.deconv4.bias.data.fill_(0)
##A3_back = net_inv.forward4_inv(A4_back)
#
print('processing A3 back to x ...')
net_inv.deconv3_reduceD.weight.data=net.conv3_reduceD.weight.data
net_inv.deconv3_reduceD.bias.data.fill_(0)
net_inv.deconv3.weight.data=net.conv3.weight.data[1:,:,:,:]
net_inv.deconv3.bias.data.fill_(0)
A2_back = net_inv.forward3_inv(A3)

print('processing A2 back to x ...')
net_inv.deconv2_reduceD.weight.data=net.conv2_reduceD.weight.data
net_inv.deconv2_reduceD.bias.data.fill_(0)
net_inv.deconv2.weight.data=net.conv2.weight.data[1:,:,:,:]
net_inv.deconv2.bias.data.fill_(0)
A1_back = net_inv.forward2_inv(A2_back)

print('processing A1 back to x ...')
net_inv.deconv1_reduceD.weight.data=net.conv1_reduceD.weight.data
net_inv.deconv1_reduceD.bias.data.fill_(0)
net_inv.deconv1.weight.data=net.conv1.weight.data[1:,:,:,:]
net_inv.deconv1.bias.data.fill_(0)
result_back = net_inv.forward1_inv(A1_back)
#print(result_back.data[0,:,:,:])
##
#
toc = time.clock()
print("Total time is %f" %(toc-tic))
showImg(result_back,1, 'result_back')
