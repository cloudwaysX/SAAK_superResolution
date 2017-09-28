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
    patchNum = ((input_width-reception_size)/stride[0]+1)**2*sampleNum
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

    ############### DC ######################
    W_pca_aranged = np.concatenate( (np.ones((1,W_pca_aranged.shape[1]))/W_pca_aranged.shape[1], W_pca_aranged) )

    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(outChannel+1,reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    
    return W_pca

def Cal_W_PCA_reduceD(X,n_keptComponent,reception_size = 1,stride = (1,1,3)):
#    print(n_keptComponent)
    assert len(X.shape)==4, "input images batch is not 4D"
    
    # reduce the DC offset
    X_AC = X[:,1:,:,:]
    
    
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
X = X[:3000]

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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
        return z1
    
    def foward1_reduceD(self,z1):
        z1_AC = z1[:,1:,:,:]; z1_DC= torch.unsqueeze(z1[:,0,:,:],dim=1)
        z1_AC = self.conv1_reduceD(z1_AC)
        z1 = torch.cat((z1_DC,z1_AC),dim = 1)
        z1 = torch.cat((z1,-z1[:,1:,:,:]),dim=1)
        A1 = F.relu(z1)
        return A1
    
    def forward2(self, A1):
        z2 = self.conv2(A1)
        return z2
    
    def foward2_reduceD(self,z2):
        z2_AC = z2[:,1:,:,:]; z2_DC= torch.unsqueeze(z2[:,0,:,:],dim=1)
        z2_AC = self.conv2_reduceD(z2_AC)
        z2 = torch.cat((z2_DC,z2_AC),dim = 1)
        z2 = torch.cat((z2,-z2[:,1:,:,:]),dim=1)
        A2 = F.relu(z2)
        return A2
    
    def forward3(self, A2):
        z3 = self.conv3(A2)
        return z3
    
    def foward3_reduceD(self,z3):
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
curInCha,curOutCha = net.conv1.weight.size()[1],net.conv1.weight.size()[0]
W_pca1= Cal_W_PCA_full(X.data.numpy(),stride = (2,2,curInCha))
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight2/l1_full_06.npy',W_pca1)
net.conv1.weight.data=torch.Tensor(W_pca1)
net.conv1.bias.data.fill_(0)
z1 = net.forward1(X)
curInCha,curOutCha = net.conv1_reduceD.weight.size()[1],net.conv1_reduceD.weight.size()[0]
W_pca1_reduceD = Cal_W_PCA_reduceD(z1.data.numpy(),curOutCha,stride = (1,1,curInCha))
np.save('/home/yifang/SAAK_superResolution/PCA_weight2/l1_reduceD_06.npy',W_pca1_reduceD)
net.conv1_reduceD.weight.data = torch.Tensor(W_pca1_reduceD)
net.conv1_reduceD.bias.data.fill_(0)
A1 = net.foward1_reduceD(z1)

print('processing A2 ...')
curInCha,curOutCha = net.conv2.weight.size()[1],net.conv2.weight.size()[0]
W_pca2= Cal_W_PCA_full(A1.data.numpy(),stride = (2,2,curInCha))
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight2/l2_full_06.npy',W_pca2)
net.conv2.weight.data=torch.Tensor(W_pca2)
net.conv2.bias.data.fill_(0)
z2 = net.forward2(A1)
curInCha,curOutCha = net.conv2_reduceD.weight.size()[1],net.conv2_reduceD.weight.size()[0]
W_pca2_reduceD = Cal_W_PCA_reduceD(z2.data.numpy(),curOutCha,stride = (1,1,curInCha))
np.save('/home/yifang/SAAK_superResolution/PCA_weight2/l2_reduceD_06.npy',W_pca2_reduceD)
net.conv2_reduceD.weight.data = torch.Tensor(W_pca2_reduceD)
net.conv2_reduceD.bias.data.fill_(0)
A2 = net.foward2_reduceD(z2)

print('processing A3 ...')
curInCha,curOutCha = net.conv3.weight.size()[1],net.conv3.weight.size()[0]
W_pca3= Cal_W_PCA_full(A2.data.numpy(),stride = (2,2,curInCha))
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight2/l3_full_06.npy',W_pca3)
net.conv3.weight.data=torch.Tensor(W_pca3)
net.conv3.bias.data.fill_(0)
z3 = net.forward3(A2)
curInCha,curOutCha = net.conv3_reduceD.weight.size()[1],net.conv3_reduceD.weight.size()[0]
W_pca3_reduceD = Cal_W_PCA_reduceD(z3.data.numpy(),curOutCha,stride = (1,1,curInCha))
np.save('/home/yifang/SAAK_superResolution/PCA_weight2/l3_reduceD_06.npy',W_pca3_reduceD)
net.conv3_reduceD.weight.data = torch.Tensor(W_pca3_reduceD)
net.conv3_reduceD.bias.data.fill_(0)
A3 = net.foward3_reduceD(z3)




