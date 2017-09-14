#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:13:01 2017

@author: yifang
"""

import numpy as np
from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows


def Cal_W_PCA(X,reception_size = 2,stride = 2):
    assert len(X.shape)==3, "input image is not 3D"
    X=np.transpose(X,(1,2,0))
    assert X.shape[0] == X.shape[1], "input image is not square"
    input_width = X.shape[0]
    input_depth = X.shape[2]
    
    #######################################
    # extracted pathes from X, the shape of X_arranged is 
    #(input_width/stride,input_width/stride,1,reception_size,reception_size,input_depth)
    X_aranged = view_as_windows(X, window_shape=(reception_size, reception_size,input_depth),step=stride)
    # rearranged the extracted patches stacks
    # shape is (n_samples, n_features)
    X_aranged = np.reshape(X_aranged, (int((input_width/stride)**2), int(reception_size**2*input_depth)))
#    print(X_aranged)
    
    #########################################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged.shape[1],svd_solver='full')
    pca.fit(X_aranged)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
#    print(W_pca_aranged)
    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(input_depth*reception_size*reception_size,reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    #######################################
    #calculate the inverse weight
    W_pca_aranged_INV =  np.linalg.inv(W_pca_aranged.T).T
    W_pca_INV = np.reshape(W_pca_aranged_INV,(input_depth*reception_size*reception_size,reception_size,reception_size,input_depth))
    W_pca_INV = np.transpose(W_pca_INV,(0,3,1,2))
    
    
    #####################################################
    # the projection of X looks like in the (sample_Num, feature_Num) shape, 
    #use for reference to test the convolution
    X_proj_reference = np.dot(X_aranged,W_pca_aranged.T)
    print(W_pca_INV.shape)
    return W_pca, W_pca_INV, X_proj_reference

X = np.arange(3*8*8,dtype=float).reshape(3,8,8)
#X=np.random.rand(3,8,8)*10
print(X)
W_pca, W_pca_INV, X_proj_reference = Cal_W_PCA(X)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 3*4, 2,stride=2 )
#        self.conv2 = nn.Conv2d(3*4*2, 3*16*2, 2,stride=2)
#        self.conv3 = nn.Conv2d(3*16*4, 3*64*4, 2,stride=2)


    def forward(self, x):
        # Max pooling over a (2, 2) window
        z1 = self.conv1(x)
        z1 = torch.cat((z1,-z1),dim=1)
        A1 = F.relu(z1)
#        z2 = self.conv2(A1)
#        z2 = torch.cat((z2,-z2),dim=1)
#        A2 = F.relu(z2)
#        z3 = self.conv2(A2)
#        z3 = torch.cat((z3,-z3),dim=1)
#        A3 = F.relu(z3)
        return A1
    
net = Net()
net.conv1.weight.data=torch.Tensor(W_pca)
X=torch.Tensor(X)
X=X.unsqueeze(0)
X=Variable(X)
result = net.forward(X)
#print(result)
#    
#    
class Inv_Net(nn.Module):

    def __init__(self):
        super(Inv_Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.deconv1 = nn.ConvTranspose2d(3*4,3,2,stride=2 )
#        self.conv2 = nn.Conv2d(3*4*2, 3*16*2, 2,stride=2)
#        self.conv3 = nn.Conv2d(3*16*4, 3*64*4, 2,stride=2)


    def forward(self, A1):
        # Max pooling over a (2, 2) window
        z1 = A1[:,0:3*4,:,:]-A1[:,3*4:3*4*2,:,:] 
        x = self.deconv1(z1)
#        z2 = torch.cat((z2,-z2),dim=1)
#        A2 = F.relu(z2)
#        z3 = self.conv2(A2)
#        z3 = torch.cat((z3,-z3),dim=1)
#        A3 = F.relu(z3)
        return x
    
net_inv = Inv_Net()
net_inv.deconv1.weight.data=torch.Tensor(W_pca)
result_back = net_inv.forward(result)
print(result_back)




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