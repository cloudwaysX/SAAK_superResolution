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


def Cal_W_PCA(X,reception_size = 2,stride = (2,2,3)):
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
#    print(X_aranged.shape)
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
    #########################################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged_whiten.shape[1],svd_solver='full')
    pca.fit(X_aranged_whiten)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
#    print(W_pca_aranged)

    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(input_depth*reception_size*reception_size,reception_size,reception_size,input_depth))
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
X = np.random.randn(1000,3,32,32)*100
print(X[0,:,:,:])
#X=np.random.rand(3,8,8)*10


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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
#W_pca, W_pca_INV, X_proj_reference, X_proj_INV_reference= Cal_W_PCA(X)
W_pca1= Cal_W_PCA(X)
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_12_3_2_2.npy',W_pca1)
# wrap input as variable
X=torch.Tensor(X)
X=Variable(X)
net.conv1.weight.data=torch.Tensor(W_pca1)
net.conv1.bias.data.fill_(0)
A1 = net.forward1(X)

print('processing A2 ...')
W_pca2= Cal_W_PCA(A1.data.numpy(),stride = (2,2,24))
#store the weight
np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_96_24_2_2.npy',W_pca2)
net.conv2.weight.data=torch.Tensor(W_pca2)
net.conv2.bias.data.fill_(0)
A2 = net.forward2(A1)

print('processing A3 ...')
W_pca3= Cal_W_PCA(A2.data.numpy(),stride = (2,2,192))
np.save('/home/yifang/SAAK_superResolution/PCA_weight/PCA_weight1_768_192_2_2.npy',W_pca3)
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

#A4_back = A4
#net_inv.deconv4.weight.data=torch.Tensor(W_pca4)
#net_inv.deconv4.bias.data.fill_(0)
#A3_back = net_inv.forward4_inv(A4_back)

A3_back=A3
print('processing A3 back to A2 ...')
net_inv.deconv3.weight.data=torch.Tensor(W_pca3)
net_inv.deconv3.bias.data.fill_(0)
A2_back = net_inv.forward3_inv(A3_back)

print('processing A2 back to A1...')
net_inv.deconv2.weight.data=torch.Tensor(W_pca2)
net_inv.deconv2.bias.data.fill_(0)
A1_back = net_inv.forward2_inv(A2_back)

print('processing A1 back to x ...')
net_inv.deconv1.weight.data=torch.Tensor(W_pca1)
net_inv.deconv1.bias.data.fill_(0)
result_back = net_inv.forward1_inv(A1_back)
print(result_back.data[0,:,:,:])
#
    

toc = time.clock()

print(toc-tic)

#import torch
#import torchvision
#import torchvision.transforms as transforms

################## Download cifar-10
#transform = transforms.Compose(
#    [transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=True, num_workers=2)
#
#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                         shuffle=False, num_workers=2)
#
#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#import matplotlib.pyplot as plt
#import numpy as np
#
## functions to show an image
#
#
#def imshow(img):
#    img = img / 2 + 0.5     # unnormalize
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    
#
#
## get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()
#
## show images
#imshow(torchvision.utils.make_grid(images))
## print labels
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))