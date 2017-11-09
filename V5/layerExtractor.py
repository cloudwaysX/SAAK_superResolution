#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 21:13:01 2017

@author: yifang
"""

    
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

class Net(nn.Module):

    def __init__(self,params,isbeforeCalssify,in_out_layers,mode):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        assert in_out_layers <= in_out_layers,'in layer must before or the same as the out layer'
        mappingWeightThreshold = params["mapping_weight_threshold"]
        zoomFactor = params['zoom factor']

        import os
        temp = in_out_layers[0]+'_2_'+in_out_layers[1]
        folder = './weight/zoom_'+str(zoomFactor)+'/'+mode+'/'+temp
        assert os.path.exists(folder),'Need to calculate weight first!Not such file as: '+folder
        

        struct_info = np.load(folder+'/'+str(mappingWeightThreshold)+'struc.npy')
        
        if struct_info[0]['in1']:
	        curL_in = struct_info[0]['in1']; curL_out = struct_info[1]['out1'];
	        self.dc1 = nn.Conv2d(curL_in, 1, zoomFactor,stride=zoomFactor ) 
	        self.upsample1 = nn.Upsample(scale_factor=zoomFactor)
	        self.conv1 = nn.Conv2d(curL_in, curL_out, zoomFactor,stride=zoomFactor ) 
        
        if struct_info[0]['in2']:
	        curL_in = struct_info[0]['in2']; curL_out = struct_info[1]['out2'];
	        self.dc2 = nn.Conv2d(curL_in, 1, zoomFactor,stride=zoomFactor ) 
	        self.upsample2 = nn.Upsample(scale_factor=zoomFactor)
	        self.conv2 = nn.Conv2d(curL_in, curL_out, zoomFactor,stride=zoomFactor ) 

        if struct_info[0]['in3']:
	        curL_in = struct_info[0]['in3']; curL_out = struct_info[1]['out3'];
	        self.dc3 = nn.Conv2d(curL_in, 1, zoomFactor,stride=zoomFactor ) 
	        self.upsample3 = nn.Upsample(scale_factor=zoomFactor)
	        self.conv3 = nn.Conv2d(curL_in, curL_out, zoomFactor,stride=zoomFactor ) 



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

def forward(dataset,params,isbeforeCalssify,datasetName="BSD_training",mode='HR',in_out_layers = ['L0','L3'], savePatch=False,printNet = False):
        # layer structure [in_layer,out_layer]
        # wrap input as variable
        X=torch.Tensor(dataset)
        X=Variable(X)

        if in_out_layers[0] == in_out_layers[1] and in_out_layers[0]=='L0':
            return X

        assert in_out_layers[0] < in_out_layers[1],'in layer must before the out layer'

        mappingWeightThresholds = params["mapping_weight_threshold"]
        zoomFactor = params['zoom factor']
        clusterI = params['cluster index']
        
        temp = in_out_layers[0]+'_2_'+in_out_layers[1]
        folder = './weight/'+'/zoom_'+str(zoomFactor)+'/'+mode+'/'+temp

        
        if savePatch and (not os.path.exists('./data/'+datasetName)):
            os.makedirs('./data/'+datasetName)
    
        net = Net(params,isbeforeCalssify,in_out_layers,mode)
        if printNet: print(net)
        
        if savePatch:
            np.save('./data/'+datasetName+'/'+mode+'_L0.npy',X.data.numpy())              
        if in_out_layers[1]=='L0':
            return X

        def L1(input):
            print('processing A1 ...')
            if isbeforeCalssify:
                W_pca1 = np.load(folder + '/L1_'+str(mappingWeightThresholds[0]) + '_beforeClassifier.npy')
            else:
                W_pca1 = np.load(folder + '/L1_'+str(mappingWeightThresholds[0]) + '_'+str(clusterI)+'.npy')
            net.conv1.weight.data=torch.Tensor(W_pca1)
            net.conv1.bias.data.fill_(0)
            net.dc1.weight.data = torch.Tensor(W_pca1[[-1],:,:,:])
            net.dc1.bias.data.fill_(0)
            A1 = net.forward1(input); 
            return A1

        def L2(input):
            print('processing A2 ...')
            if isbeforeCalssify:
                W_pca2 = np.load(folder + '/L2_'+str(mappingWeightThresholds[1]) + '_beforeClassifier.npy')
            else:
                W_pca2 = np.load(folder + '/L2_'+str(mappingWeightThresholds[1]) + '_'+str(clusterI)+'.npy')
            net.conv2.weight.data=torch.Tensor(W_pca2)
            net.conv2.bias.data.fill_(0)
            net.dc2.weight.data = torch.Tensor(W_pca2[[-1],:,:,:])
            net.dc2.bias.data.fill_(0)
            A2 = net.forward2(input);
            return A2

        def L3(input):
            print('processing A3 ...')
            if isbeforeCalssify:
                W_pca3 = np.load(folder + '/L3_'+str(mappingWeightThresholds[1]) + '_beforeClassifier.npy')
            else:
                W_pca3 = np.load(folder + '/L3_'+str(mappingWeightThresholds[1]) + '_'+str(clusterI)+'.npy')
            net.conv3.weight.data=torch.Tensor(W_pca3)
            net.conv3.bias.data.fill_(0)
            net.dc3.weight.data = torch.Tensor(W_pca3[[-1],:,:,:])
            net.dc3.bias.data.fill_(0)
            A3 = net.forward3(input);
            return A3


        if in_out_layers[1]=='L1':
            return L1(X)
        elif in_out_layers[1] == 'L2':
            if in_out_layers[0] == 'L0': 
                A1 = L1(X); return L2(A1)
            else:
                return L2(X)
        else:
            if in_out_layers[0] == 'L0': 
                A1 = L1(X); del X
                A2 = L2(A1); del A1
                return L3(A2)
            elif in_out_layers[0] == 'L1':
                A2 = L2(X); del X
                return L3(A2)
            else:
                return L3(X)

class Inv_Net(nn.Module):

    def __init__(self,params,isbeforeCalssify,in_out_layers,mode):
        super(Inv_Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        assert ord(in_out_layers[0][1]) < ord(in_out_layers[1][1]) or in_out_layers[1]=='L0','in layer must before the out layer'
        mappingWeightThreshold = params["mapping_weight_threshold"]
        zoomFactor = params['zoom factor']

        import os
        temp = in_out_layers[0]+'_2_'+in_out_layers[1]
        folder = './weight/zoom_'+str(zoomFactor)+'/'+mode+'/'+temp
        assert os.path.exists(folder),'Need to calculate weight first!Not such file as: '+folder
        

        struct_info = np.load(folder+'/'+str(mappingWeightThreshold)+'struc.npy')


        if struct_info[0]['in1']:
            curL_in = struct_info[0]['in1']; curL_out = struct_info[1]['out1'];
            self.deconv1 = nn.ConvTranspose2d(curL_out,curL_in,zoomFactor,stride=zoomFactor )
            self.upsample1 = nn.Upsample(scale_factor=zoomFactor)

        if struct_info[0]['in2']:
            curL_in = struct_info[0]['in2']; curL_out = struct_info[1]['out2'];
            self.deconv2 = nn.ConvTranspose2d(curL_out,curL_in,zoomFactor,stride=zoomFactor )
            self.upsample2 = nn.Upsample(scale_factor=zoomFactor)

        if struct_info[0]['in3']:
            curL_in = struct_info[0]['in3']; curL_out = struct_info[1]['out3'];
            self.deconv3 = nn.ConvTranspose2d(curL_out,curL_in,zoomFactor,stride=zoomFactor )
            self.upsample3 = nn.Upsample(scale_factor=zoomFactor)
        

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

def inverse(forward_result,params,isbeforeCalssify,in_out_layers, mode='HR',printNet = False):
        mappingWeightThresholds = params["mapping_weight_threshold"]
        zoomFactor = params['zoom factor']
        clusterI = params['cluster index']
        
        temp = in_out_layers[0]+'_2_'+in_out_layers[1]
        folder = './weight/'+'/zoom_'+str(zoomFactor)+'/'+mode+'/'+temp

        assert os.path.exists(folder),'Need to calculate weight first! Cannot find: '+folder

        net_inv = Inv_Net(params,isbeforeCalssify,in_out_layers,mode)

        def L1(input):
            print('processing A1 back to x ...')
            if isbeforeCalssify:
                W_pca = np.load(folder + '/L1_'+str(mappingWeightThresholds[0]) + '_beforeClassifier.npy')
            else:
                W_pca = np.load(folder + '/L1_'+str(mappingWeightThresholds[0]) + '_'+str(clusterI)+'.npy')

            net_inv.deconv1.weight.data = torch.Tensor(W_pca)
            net_inv.deconv1.bias.data.fill_(0)
            return net_inv.forward1_inv(input)

        def L2(input):
            print('processing A2 back to x ...')
            if isbeforeCalssify:
                W_pca = np.load(folder + '/L2_'+str(mappingWeightThresholds[1]) + '_beforeClassifier.npy')
            else:
                W_pca = np.load(folder + '/L2_'+str(mappingWeightThresholds[1]) + '_'+str(clusterI)+'.npy')

            net_inv.deconv2.weight.data = torch.Tensor(W_pca)
            net_inv.deconv2.bias.data.fill_(0)
            return net_inv.forward2_inv(input)

        def L3(input):
            print('processing A3 back to x ...')
            if isbeforeCalssify:
                W_pca = np.load(folder + '/L3_'+str(mappingWeightThresholds[2]) + '_beforeClassifier.npy')
            else:
                W_pca = np.load(folder + '/L3_'+str(mappingWeightThresholds[2]) + '_'+str(clusterI)+'.npy')

            net_inv.deconv3.weight.data = torch.Tensor(W_pca)
            net_inv.deconv3.bias.data.fill_(0)
            return net_inv.forward3_inv(input)


        if in_out_layers[1]=='L1':
            return L1(forward_result)
        elif in_out_layers[1] == 'L2':
            if in_out_layers[0] == 'L0': 
                A1 = L2(forward_result); del forward_result
                return L1(A1)
            else:
                return L2(forward_result)
        else:
            if in_out_layers[0] == 'L0': 
                A2 = L3(forward_result); del forward_result
                A1 = L2(A1); del A1
                return L1(A1)
            elif in_out_layers[0] == 'L1':
                A2 = L3(forward_result); del forward_result
                return L2(A2)
            else:
                return L3(forward_result)
