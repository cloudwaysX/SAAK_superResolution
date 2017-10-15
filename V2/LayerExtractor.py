#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:18:39 2017

@author: yifang
"""

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



#X = np.arange(3*8*8,dtype=float).reshape(3,8,8)
#X = np.stack((X,X))
import os
from scipy import misc
import matplotlib.pyplot as plt


def readHRImg(path,name):
    img= misc.imread(os.path.join(path,name), flatten= 0)
    img_resized = misc.imresize( img, (32,32,3),interp = 'cubic')
    return img_resized

def readLRImg(path,name):
    img= misc.imread(os.path.join(path,name), flatten= 0)
    img_resized = misc.imresize( img, (8,8,3),interp = 'cubic')
    img_resized = misc.imresize( img_resized, (32,32,3),interp = 'cubic')
    return img_resized

def showImg(outBatch,index=0,title="inverse Rsult"):
    # input: outBatch with size (imageNum, imageDepth, ImageW, ImageH)
    # index = the index of the image you want to show; tiile = string of plot title
    displayImg = outBatch[index,:,:,:].data.numpy()
    displayImg=np.uint8(displayImg)
    plt.figure()    
    plt.title(title)
    if displayImg.shape[0]==3:
        plt.imshow(np.transpose(displayImg, (1,2,0)))
    else:
        import matplotlib as mpl
        plt.imshow(displayImg[0],cmap=mpl.cm.Greys)
    plt.show()
    plt.figure()   

def showFeatureVec(outFeatureVec,index=0,title = "Feature Vec"):
    from pylab import stem
    x_axis = range(0,outFeatureVec.size()[1])
    rowNum = outFeatureVec.size()[2]; colNum=outFeatureVec.size()[3]
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
        z3_AC = self.conv3_reduceD(z3_AC)
        z3_DC = self.downsample3(z3_DC)
        A3_1 = F.relu(z3_AC);A3_2=F.relu(-z3_AC)
        A3 = torch.cat((z3_DC,A3_1,A3_2),dim = 1)
        return A3


def forward(dataset,datasetName="training",mode='HR',layer = 'L1'):
        import scipy.io    
    
        net = Net()
        # wrap input as variable
        X=torch.Tensor(dataset[mode])
        X=Variable(X)
        print('processing A1 ...')
        W_pca1 = np.load('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L1full.npy')
        net.conv1.weight.data=torch.Tensor(W_pca1)
        net.conv1.bias.data.fill_(0)
        W_pca1_reduceD = np.load('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L1reduceD.npy')
        net.conv1_reduceD.weight.data = torch.Tensor(W_pca1_reduceD)
        net.conv1_reduceD.bias.data.fill_(0)
        A1 = net.forward1(X); 
        scipy.io.savemat('./data/'+datasetName+'/MNIST_'+mode+'_L1.mat', mdict={'MNIST_'+mode+'_L1': np.transpose(A1.data.numpy(),(2,3,1,0))})  

        if layer=='L1':
            return net,A1
        
        print('processing A2 ...')
        W_pca2 = np.load('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L2full.npy')
        net.conv2.weight.data=torch.Tensor(W_pca2)
        net.conv2.bias.data.fill_(0)
        W_pca2_reduceD = np.load('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L2reduceD.npy')
        net.conv2_reduceD.weight.data = torch.Tensor(W_pca2_reduceD)
        net.conv2_reduceD.bias.data.fill_(0)
        A2 = net.forward2(A1);del A1
        scipy.io.savemat('./data/'+datasetName+'/MNIST_'+mode+'_L2.mat', mdict={'MNIST_'+mode+'_L2': np.transpose(A2.data.numpy(),(2,3,1,0))})          

        if layer=='L2':
            return net,A2
        
        print('processing A3 ...')
        W_pca3 = np.load('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L3full.npy')
        net.conv3.weight.data=torch.Tensor(W_pca3)
        net.conv3.bias.data.fill_(0)
        W_pca3_reduceD = np.load('/home/yifang/SAAK_superResolution/V2/weight/'+mode+'/MNIST_L3reduceD.npy')
        net.conv3_reduceD.weight.data = torch.Tensor(W_pca3_reduceD)
        net.conv3_reduceD.bias.data.fill_(0)
        A3 = net.forward3(A2); del A2
        scipy.io.savemat('./data/'+datasetName+'/MNIST_'+mode+'_L3.mat', mdict={'MNIST_'+mode+'_L3': np.transpose(A3.data.numpy(),(2,3,1,0))})          

        
        return net, A3
#
#showFeatureVec(A3)
#
#
class Inv_Net(nn.Module):

    def __init__(self,net=Net()):
        super(Inv_Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        curL_in,curL_out = net.conv1_reduceD.weight.size()[0],net.conv1_reduceD.weight.size()[1];
        self.deconv1_reduceD = nn.ConvTranspose2d(curL_in,curL_out,1,stride=1 )
        curL_in,curL_out = net.conv1.weight.size()[0],net.conv1.weight.size()[1];
        self.deconv1 = nn.ConvTranspose2d(curL_in,curL_out,2,stride=2 )
        self.upsample1 = nn.Upsample(scale_factor=2)
        
        curL_in,curL_out = net.conv2_reduceD.weight.size()[0],net.conv2_reduceD.weight.size()[1];
        self.deconv2_reduceD = nn.ConvTranspose2d(curL_in,curL_out,1,stride=1 )
        curL_in,curL_out = net.conv2.weight.size()[0],net.conv2.weight.size()[1];
        self.deconv2 = nn.ConvTranspose2d(curL_in-1,curL_out,2,stride=2 )
        self.upsample2 = nn.Upsample(scale_factor=2)
#        
        curL_in,curL_out = net.conv3_reduceD.weight.size()[0],net.conv3_reduceD.weight.size()[1];
        self.deconv3_reduceD = nn.ConvTranspose2d(curL_in,curL_out,1,stride=1 )
        curL_in,curL_out = net.conv3.weight.size()[0],net.conv3.weight.size()[1];
        self.deconv3 = nn.ConvTranspose2d(curL_in-1,curL_out,2,stride=2 )
        self.upsample3 = nn.Upsample(scale_factor=2)

    def forward1_inv(self, A1):
        half=(A1.size()[1]-1)/2;half = int(half)
        z1 = A1[:,1:half+1,:,:]-A1[:,half+1:,:,:] 
        z1 = self.deconv1_reduceD(z1)
        x = self.deconv1(z1)
        x = x+self.upsample1(torch.unsqueeze(A1[:,0,:,:],dim=1))
        return x
    
    def forward2_inv(self, A2):
        half=(A2.size()[1]-1)/2;half = int(half)
        z2 = A2[:,1:half+1,:,:]-A2[:,half+1:,:,:] 
        z2 = self.deconv2_reduceD(z2)
        A1 = self.deconv2(z2)
        A1 = A1+self.upsample2(torch.unsqueeze(A2[:,0,:,:],dim=1))
        return A1
        
    def forward3_inv(self, A3):
        half=(A3.size()[1]-1)/2;half = int(half)
        z3 = A3[:,1:half+1,:,:]-A3[:,half+1:,:,:] 
        z3 = self.deconv3_reduceD(z3)
        A2 = self.deconv3(z3)
        A2 = A2+self.upsample3(torch.unsqueeze(A3[:,0,:,:],dim=1))
        return A2
    
def inverse(forward_result,net,layer = 'L1'):
    net_inv = Inv_Net()
    
    
    if layer == 'L3':
        A3_back = forward_result
        print('processing A3 back to x ...')
        net_inv.deconv3_reduceD.weight.data=net.conv3_reduceD.weight.data
        net_inv.deconv3_reduceD.bias.data.fill_(0)
        net_inv.deconv3.weight.data=net.conv3.weight.data
        net_inv.deconv3.bias.data.fill_(0)
        A2_back = net_inv.forward3_inv(A3_back)
        #
        ##A2_back=A2
        print('processing A2 back to x ...')
        net_inv.deconv2_reduceD.weight.data=net.conv2_reduceD.weight.data
        net_inv.deconv2_reduceD.bias.data.fill_(0)
        net_inv.deconv2.weight.data=net.conv2.weight.data
        net_inv.deconv2.bias.data.fill_(0)
        A1_back = net_inv.forward2_inv(A2_back)
        
        print('processing A1 back to x ...')
        net_inv.deconv1_reduceD.weight.data=net.conv1_reduceD.weight.data
        net_inv.deconv1_reduceD.bias.data.fill_(0)
        net_inv.deconv1.weight.data=net.conv1.weight.data
        net_inv.deconv1.bias.data.fill_(0)
        result_back = net_inv.forward1_inv(A1_back)
    elif layer == 'L2':
        A2_back=forward_result
        print('processing A2 back to x ...')
        net_inv.deconv2_reduceD.weight.data=net.conv2_reduceD.weight.data
        net_inv.deconv2_reduceD.bias.data.fill_(0)
        net_inv.deconv2.weight.data=net.conv2.weight.data
        net_inv.deconv2.bias.data.fill_(0)
        A1_back = net_inv.forward2_inv(A2_back)
        
        print('processing A1 back to x ...')
        net_inv.deconv1_reduceD.weight.data=net.conv1_reduceD.weight.data
        net_inv.deconv1_reduceD.bias.data.fill_(0)
        net_inv.deconv1.weight.data=net.conv1.weight.data
        net_inv.deconv1.bias.data.fill_(0)
        result_back = net_inv.forward1_inv(A1_back)
    else:
        A1_back = forward_result
        print('processing A1 back to x ...')
        net_inv.deconv1_reduceD.weight.data=net.conv1_reduceD.weight.data
        net_inv.deconv1_reduceD.bias.data.fill_(0)
        net_inv.deconv1.weight.data=net.conv1.weight.data
        net_inv.deconv1.bias.data.fill_(0)
        result_back = net_inv.forward1_inv(A1_back)
        
    return result_back

#    showImg(result_back,1, 'result_back')