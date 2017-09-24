#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:13:01 2017

@author: yifang
"""


import time
tic = time.clock()

import numpy as np
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
from sklearn import preprocessing

def CalcNextChannelNum(preOutNum, keepComp=1):
    nextL_in = preOutNum*2-1
    nextL_out = int(round(nextL_in*4*keepComp))+1
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
    patchNum = ((input_width-reception_size)/stride[0]+1)**2*sampleNum
    featureNum = reception_size**2*input_depth
    X_aranged = np.reshape(X_aranged, (int(patchNum), int(featureNum)))
    
# =============================================================================
#     #######################################
#     #standardlization before applying PCA
#     X_aranged_whiten=preprocessing.scale(X_aranged) 
#     X_aranged_mean = np.mean (X_aranged,axis = 0)
#     print(X_aranged_mean)
#     
# #    print(X_aranged_whiten)
#
# =============================================================================
    X_aranged_whiten = X_aranged
    ############### AC ####################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged_whiten.shape[1],svd_solver='full')
    pca.fit(X_aranged_whiten)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
    W_pca_aranged = W_pca_aranged[:n_keptComponent]

    ############### DC ######################
    W_pca_aranged = np.concatenate( (np.ones((1,W_pca_aranged.shape[1]))/W_pca_aranged.shape[1], W_pca_aranged) )

    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(n_keptComponent+1,reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    
#     No need for PCA because inverse = transpose
# =============================================================================
#     #######################################
#     #calculate the inverse weight
#     W_pca_aranged_INV =  np.linalg.inv(W_pca_aranged.T).T
#     W_pca_INV = np.reshape(W_pca_aranged_INV,(input_depth*reception_size*reception_size,reception_size,reception_size,input_depth))
#     W_pca_INV = np.transpose(W_pca_INV,(0,3,1,2))
# =============================================================================
    
    
#     Debugging purpose only
# =============================================================================
#     #####################################################
#     # the projection of X looks like in the (sample_Num, feature_Num) shape, 
#     #use for reference to test the convolution
#     X_proj_reference = np.dot(X_aranged,W_pca_aranged.T)
#     X_proj_INV_reference = np.dot(X_proj_reference,W_pca_aranged_INV.T)
# #    return W_pca, W_pca_INV, X_proj_reference, X_proj_INV_reference
# =============================================================================
    return W_pca



#X = np.arange(3*8*8,dtype=float).reshape(3,8,8)
#X = np.stack((X,X))
#import os
#from scipy import misc
#import matplotlib.pyplot as plt
#
#
#def readImg(path,name):
#    image= misc.imread(os.path.join(path,name), flatten= 0)
#    return image
#
#path = '/home/yifang/SAAK_superResolution/SRCNN/Set5'
#name = 'baby_GT.bmp'
#
#img = readImg(path,name)
#
#img_resized = misc.imresize( img, (32,32,3),interp = 'cubic')
#X = np.tile(img_resized,(1000,1,1,1))
#X = np.transpose(X, (0,3,1,2))
#X = np.float32(X)
def load_databatch(data_folder, idx, img_size=32):
    import os
    import pickle
    import numpy as np
    import lasagne
    
    def unpickle(name):
        with open(name, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
            return content
    
    
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    x = x/np.float32(255)
    mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        mean=mean_image)

data_folder = '/home/yifang/SAAK_superResolution/data/train'
the_batch = load_databatch(data_folder,1)
X=the_batch['X_train']+np.reshape(the_batch['mean'],(3,32,32))
X = X[:1000]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        keepComp_init = 0.4
        curL_in = 3; curL_out = int(3*4*keepComp_init) + 1 # initial
        self.conv1 = nn.Conv2d(curL_in, curL_out, 2,stride=2 ) #full 3,13
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=0.4)
        self.conv2 = nn.Conv2d(curL_in, curL_out, 2,stride=2)#full 24+1, 100+1
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=0.4)
        self.conv3 = nn.Conv2d(curL_in, curL_out, 2,stride=2)#full 200+1, 804+1
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=0.4)
        self.conv4 = nn.Conv2d(curL_in, curL_out, 2,stride=2)#full 804*2+1, (804*2+1)*4+1
        curL_in,curL_out = CalcNextChannelNum(curL_out,keepComp=0.4)
        self.conv5 = nn.Conv2d(curL_in, curL_out, 2,stride=2) # full 804*2+1, (804*2+1)*4+1

    def forward1(self, x):
        z1 = self.conv1(x)
        z1 = torch.cat((z1,-z1[:,1:,:,:]),dim=1)
        A1 = F.relu(z1)
        return A1
    
    def forward2(self,A1):
        z2 = self.conv2(A1)
        z2 = torch.cat((z2,-z2[:,1:,:,:]),dim=1)
        A2 = F.relu(z2)
        return A2
        
    def forward3(self,A2):
        z3 = self.conv3(A2)
        z3 = torch.cat((z3,-z3[:,1:,:,:]),dim=1)
        A3 = F.relu(z3)
        return A3
    
    def forward4(self,A3):
        z4 = self.conv4(A3)
        z4 = torch.cat((z4,-z4[:,1:,:,:]),dim=1)
        A4 = F.relu(z4)
        A4 = np.cat((z4[:,0,:,:],A4),dim=1)
        return A4
    
    def forward5(self,A4):
        z5 = self.conv5(A4)
        z5 = torch.cat((z5,-z5[:,1:,:,:]),dim=1)
        A5 = F.relu(z5)
        return A5

    
    
net = Net()
# wrap input as variable
X=torch.Tensor(X)
X=Variable(X)
print('processing A1 ...')
#W_pca, W_pca_INV, X_proj_reference, X_proj_INV_reference= Cal_W_PCA(X)
curInCha,curOutCha = net.conv1.weight.size()[1],net.conv1.weight.size()[0]
W_pca1= Cal_W_PCA(X.data.numpy(),curOutCha-1,stride = (2,2,curInCha))
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_12_3_2_2_04.npy',W_pca1)
net.conv1.weight.data=torch.Tensor(W_pca1)
net.conv1.bias.data.fill_(0)
A1 = net.forward1(X)

print('processing A2 ...')
curInCha,curOutCha = net.conv2.weight.size()[1],net.conv2.weight.size()[0]
W_pca2= Cal_W_PCA(A1.data.numpy(),curOutCha-1,stride = (2,2,curInCha))
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_96_24_2_2_04.npy',W_pca2)
net.conv2.weight.data=torch.Tensor(W_pca2)
net.conv2.bias.data.fill_(0)
A2 = net.forward2(A1)

print('processing A3 ...')
curInCha,curOutCha = net.conv3.weight.size()[1],net.conv3.weight.size()[0]
W_pca3= Cal_W_PCA(A2.data.numpy(),curOutCha-1,stride = (2,2,curInCha))
np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_768_192_2_2_04.npy',W_pca3)
net.conv3.weight.data=torch.Tensor(W_pca3)
net.conv3.bias.data.fill_(0)
A3 = net.forward3(A2)

#print('processing A4 ...')
#curInCha,curOutCha = net.conv4.weight.size()[1],net.conv4.weight.size()[0]
#W_pca4= Cal_W_PCA(X.data.numpy(),curOutCha-1,stride = (2,2,curInCha))
#net.conv4.weight.data=torch.Tensor(W_pca4)
#net.conv4.bias.data.fill_(0)
#A4 = net.forward4(A3)





