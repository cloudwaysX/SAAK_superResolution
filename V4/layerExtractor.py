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
#from scipy import misc
import matplotlib.pyplot as plt


#def readHRImg(path,name):
#    img= misc.imread(os.path.join(path,name), flatten= 0)
#    img_resized = misc.imresize( img, (32,32,3),interp = 'cubic')
#    return img_resized
#
#def readLRImg(path,name):
#    img= misc.imread(os.path.join(path,name), flatten= 0)
#    img_resized = misc.imresize( img, (8,8,3),interp = 'cubic')
#    img_resized = misc.imresize( img_resized, (32,32,3),interp = 'cubic')
#    return img_resized
#
def showImg(outBatch,index=0,title="inverse Result",lowThreshold = 0):
    # input: outBatch with size (imageNum, imageDepth, ImageW, ImageH)
    # index = the index of the image you want to show; tiile = string of plot title
    displayImg = outBatch[index,:,:,:].data.numpy()
    displayImg[displayImg<lowThreshold]=0
    displayImg=np.uint8(np.clip(displayImg,0,255))
    plt.figure()    
    plt.title(title)
    if displayImg.shape[0]==3:
        plt.imshow(np.transpose(displayImg, (1,2,0)))
    else:
        import matplotlib as mpl
        plt.imshow(displayImg[0],cmap=mpl.cm.Greys)
    plt.show()

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
            stem(x_axis, outFeatureVec[index,:,row,col].data.numpy())
    
    
import numpy as np


#def CalcNextChannelNum(preOutNum, keepComp=1):
#    nextL_in = preOutNum*2 + 1 -2
#    nextL_out = round(nextL_in*4*keepComp)
#    return nextL_in,nextL_out

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,weight_thresholds,mode,folder,zoomfactor):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        struct_info = np.load('./'+folder+'/'+mode+'/'+str(weight_thresholds[0])+'_'+str(weight_thresholds[1])+'_'+\
            str(weight_thresholds[2])+'_struc.npy')
        
        
        curL_in = struct_info[0]['in1']; curL_out = struct_info[1]['out1'];
        self.dc1 = nn.Conv2d(curL_in, 1, zoomfactor,stride=zoomfactor ) 
        self.upsample1 = nn.Upsample(scale_factor=zoomfactor)
        self.conv1 = nn.Conv2d(curL_in, curL_out, zoomfactor,stride=zoomfactor ) 
        
        curL_in = struct_info[0]['in2']; curL_out = struct_info[1]['out2'];
        self.dc2 = nn.Conv2d(curL_in, 1, zoomfactor,stride=zoomfactor ) 
        self.upsample2 = nn.Upsample(scale_factor=zoomfactor)
        self.conv2 = nn.Conv2d(curL_in, curL_out, zoomfactor,stride=zoomfactor ) 

        
        curL_in = struct_info[0]['in3']; curL_out = struct_info[1]['out3'];
        self.dc3 = nn.Conv2d(curL_in, 1, zoomfactor,stride=zoomfactor ) 
        self.upsample3 = nn.Upsample(scale_factor=zoomfactor)
        self.conv3 = nn.Conv2d(curL_in, curL_out, zoomfactor,stride=zoomfactor ) 



    def forward1(self, x):
        z1_DC = self.dc1(x)
        n_feature = x.size()[1]*2*2
        z1_mean = self.upsample1(z1_DC/np.sqrt(n_feature))
        z1_AC = self.conv1(x-z1_mean)
        z1_AC = z1_AC[:,:-1,:,:]
        A1_1 = F.relu(z1_AC);A1_2=F.relu(-z1_AC)
        A1 = torch.cat((z1_DC,A1_1,A1_2),dim = 1)
        return A1
    
    def forward2(self, A1):
        z2_DC = self.dc2(A1)
        n_feature = A1.size()[1]*2*2
        z2_mean = self.upsample1(z2_DC/np.sqrt(n_feature))
        z2_AC = self.conv2(A1-z2_mean)
        z2_AC = z2_AC[:,:-1,:,:]
        A2_1 = F.relu(z2_AC);A2_2=F.relu(-z2_AC)
        A2 = torch.cat((z2_DC,A2_1,A2_2),dim = 1)
        return A2
    
    def forward3(self, A2):
        z3_DC = self.dc3(A2)
        n_feature = A2.size()[1]*2*2
        z3_mean = self.upsample1(z3_DC/np.sqrt(n_feature))
        z3_AC = self.conv3(A2-z3_mean)
        z3_AC = z3_AC[:,:-1,:,:]
        A3_1 = F.relu(z3_AC);A3_2=F.relu(-z3_AC)
        A3 = torch.cat((z3_DC,A3_1,A3_2),dim = 1)
        return A3

def forward(dataset,weight_thresholds = [0,0,0],datasetName="MNIST_training",mode='HR',layer = 'L1',savePatch=False,folder='weight',zoomfactor=2,printNet = False):
        import scipy.io   
        
        folder = folder + '/zoom_'+str(zoomfactor)
        if savePatch and (not os.path.exists('./data/'+datasetName)):
            os.makedirs('./data/'+datasetName)
    
        net = Net(weight_thresholds,mode,folder,zoomfactor)
        if printNet: print(net)
        # wrap input as variable
        X=torch.Tensor(dataset[mode])
        X=Variable(X)
        
        if savePatch:
            scipy.io.savemat('./data/'+datasetName+'/'+mode+'_L0.mat', mdict={mode+'_L0': np.transpose(X.data.numpy(),(2,3,1,0))})
            np.save('./data/'+datasetName+'/'+mode+'_L0.npy',X.data.numpy())              
        if layer=='L0':
            return X

        
        print('processing A1 ...')
        W_pca1 = np.load('./'+folder+'/'+mode+'/_L1_'+str(weight_thresholds[0])+'.npy')
        net.conv1.weight.data=torch.Tensor(W_pca1)
        net.conv1.bias.data.fill_(0)
        net.dc1.weight.data = torch.Tensor(W_pca1[[-1],:,:,:])
        net.dc1.bias.data.fill_(0)
        A1 = net.forward1(X); 
        if savePatch:
            scipy.io.savemat('./data/'+datasetName+'/'+mode+'_L1_'+str(weight_thresholds[0])+'.mat', mdict={mode+'_L1': \
                             np.transpose(A1.data.numpy(),(2,3,1,0))})  
            np.save('./data/'+datasetName+'/'+mode+'_L1_'+str(weight_thresholds[0])+'.npy',A1.data.numpy())     
        if layer=='L1':
            return A1
        
        print('processing A2 ...')
        W_pca2 = np.load('./'+folder+'/'+mode+'/_L2_'+str(weight_thresholds[1])+'.npy')
        net.conv2.weight.data=torch.Tensor(W_pca2)
        net.conv2.bias.data.fill_(0)
        net.dc2.weight.data = torch.Tensor(W_pca2[[-1],:,:,:])
        net.dc2.bias.data.fill_(0)
        A2 = net.forward2(A1);del A1
        if savePatch:
            scipy.io.savemat('./data/'+datasetName+'/'+mode+'_L2_'+str(weight_thresholds[1])+'.mat', mdict={mode+'_L2': \
                             np.transpose(A2.data.numpy(),(2,3,1,0))})  
            np.save('./data/'+datasetName+'/'+mode+'_L2_'+str(weight_thresholds[1])+'.npy',A2.data.numpy())   
        if layer=='L2':
            return A2
        
        print('processing A3 ...')
        W_pca3 = np.load('./'+folder+'/'+mode+'/_L3_'+str(weight_thresholds[2])+'.npy')
        net.conv3.weight.data=torch.Tensor(W_pca3)
        net.conv3.bias.data.fill_(0)
        net.dc3.weight.data = torch.Tensor(W_pca3[[-1],:,:,:])
        net.dc3.bias.data.fill_(0)
        A3 = net.forward3(A2); del A2
        if savePatch:
            scipy.io.savemat('./data/'+datasetName+'/'+mode+'_L3_'+str(weight_thresholds[2])+'.mat', mdict={mode+'_L3': \
                             np.transpose(A3.data.numpy(),(2,3,1,0))})  
            np.save('./data/'+datasetName+'/'+mode+'_L3_'+str(weight_thresholds[2])+'.npy',A3.data.numpy())   
        return A3
#
#showFeatureVec(A3)
#
#
class Inv_Net(nn.Module):

    def __init__(self,weight_thresholds,mode,folder,zoomfactor):
        super(Inv_Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        struct_info = np.load('./'+folder+'/'+mode+'/'+str(weight_thresholds[0])+'_'+str(weight_thresholds[1])+'_'+\
            str(weight_thresholds[2])+'_struc.npy')

        curL_in = struct_info[0]['in1']; curL_out = struct_info[1]['out1'];
        self.deconv1 = nn.ConvTranspose2d(curL_out,curL_in,zoomfactor,stride=zoomfactor )
        self.upsample1 = nn.Upsample(scale_factor=zoomfactor)

        curL_in = struct_info[0]['in2']; curL_out = struct_info[1]['out2'];
        self.deconv2 = nn.ConvTranspose2d(curL_out,curL_in,zoomfactor,stride=zoomfactor )
        self.upsample2 = nn.Upsample(scale_factor=zoomfactor)

        curL_in = struct_info[0]['in3']; curL_out = struct_info[1]['out3'];
        self.deconv3 = nn.ConvTranspose2d(curL_out,curL_in,zoomfactor,stride=zoomfactor )
        self.upsample3 = nn.Upsample(scale_factor=zoomfactor)
        

    def forward1_inv(self, A1):
        half=(A1.size()[1]-1)/2;half = int(half)
        z1 = A1[:,1:half+1,:,:]-A1[:,half+1:,:,:] 
        zero_mean = torch.zeros(z1.size())
        zero_mean = zero_mean[:,[0],:,:]
        z1 = torch.cat((z1,zero_mean),dim=1)
        n_feature = z1.size()[1]
        x_AC = self.deconv1(z1)
        x = x_AC+self.upsample1(torch.unsqueeze(A1[:,0,:,:]/np.sqrt(n_feature),dim=1))
        return x
    
    def forward2_inv(self, A2):
        half=(A2.size()[1]-1)/2;half = int(half)
        z2 = A2[:,1:half+1,:,:]-A2[:,half+1:,:,:] 
        zero_mean = torch.zeros(z2.size())
        zero_mean = zero_mean[:,[0],:,:]
        z2 = torch.cat((z2,zero_mean),dim=1)
        n_feature = z2.size()[1]
        A1_AC = self.deconv2(z2)
        A1 = A1_AC+self.upsample2(torch.unsqueeze(A2[:,0,:,:]/np.sqrt(n_feature),dim=1))
        return A1
        
    def forward3_inv(self, A3):
        half=(A3.size()[1]-1)/2;half = int(half)
        z3 = A3[:,1:half+1,:,:]-A3[:,half+1:,:,:] 
        zero_mean = torch.zeros(z3.size())
        zero_mean = zero_mean[:,[0],:,:]
        z3 = torch.cat((z3,zero_mean),dim=1)
        n_feature = z3.size()[1]
        A2_AC = self.deconv3(z3)
        A2 = A2_AC+self.upsample3(torch.unsqueeze(A3[:,0,:,:]/np.sqrt(n_feature),dim=1))
        return A2
    
def inverse(forward_result,weight_thresholds = [0,0,0],mode='HR',layer = 'L1',folder='weight',zoomfactor=2, printNet = False):
    
    if layer =='L0':
        return forward_result
      
    folder = folder + '/zoom_'+str(zoomfactor)
    net_inv = Inv_Net(weight_thresholds,mode,folder,zoomfactor)
    if printNet: print(net_inv)
    
    
    if layer == 'L3':
        A3_back = forward_result
        print('processing A3 back to x ...')
        net_inv.deconv3.weight.data=torch.Tensor(np.load('./'+folder+'/'+mode+'/_L3_'+str(weight_thresholds[2])+'.npy'))
        net_inv.deconv3.bias.data.fill_(0)
        A2_back = net_inv.forward3_inv(A3_back);del A3_back

        print('processing A2 back to x ...')
        net_inv.deconv2.weight.data=torch.Tensor(np.load('./'+folder+'/'+mode+'/_L2_'+str(weight_thresholds[1])+'.npy'))
        net_inv.deconv2.bias.data.fill_(0)
        A1_back = net_inv.forward2_inv(A2_back);del A2_back
        
        print('processing A1 back to x ...')
        net_inv.deconv1.weight.data=torch.Tensor(np.load('./'+folder+'/'+mode+'/_L1_'+str(weight_thresholds[0])+'.npy'))
        net_inv.deconv1.bias.data.fill_(0)
        result_back = net_inv.forward1_inv(A1_back);del A1_back
    elif layer == 'L2':
        A2_back=forward_result
        print('processing A2 back to x ...')
        net_inv.deconv2.weight.data=torch.Tensor(np.load('./'+folder+'/'+mode+'/_L2_'+str(weight_thresholds[1])+'.npy'))
        net_inv.deconv2.bias.data.fill_(0)
        A1_back = net_inv.forward2_inv(A2_back);del A2_back
        
        print('processing A1 back to x ...')
        net_inv.deconv1.weight.data=torch.Tensor(np.load('./'+folder+'/'+mode+'/_L1_'+str(weight_thresholds[0])+'.npy'))
        net_inv.deconv1.bias.data.fill_(0)
        result_back = net_inv.forward1_inv(A1_back);del A1_back
    else:
        A1_back = forward_result
        print('processing A1 back to x ...')
        net_inv.deconv1.weight.data=torch.Tensor(np.load('./'+folder+'/'+mode+'/_L1_'+str(weight_thresholds[0])+'.npy'))
        net_inv.deconv1.bias.data.fill_(0)
        result_back = net_inv.forward1_inv(A1_back);del A1_back
        
    return result_back