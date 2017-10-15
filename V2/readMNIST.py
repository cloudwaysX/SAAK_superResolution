#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:54:37 2017

@author: yifang
"""

import os
import struct
import numpy as np

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
class DatasetMNIST():
    
    def __init__(self):
        img_HR = None
        img_LR = None
        #shape is (size,size,1 sample number)
        self.trainset = {'HR':img_HR,'LR':img_LR} 
        self.testset = {'HR':img_HR,'LR':img_LR}
        
    def readMNIST(self,dataset = "training", path = ".",scale = 4):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """
        from scipy import misc
        if dataset is "training":
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            sampleNum = 60000
        elif dataset is "testing":
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            sampleNum = 10000
        else:
            raise(ValueError, "dataset must be 'testing' or 'training'")
    #
        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(sampleNum, rows, cols)
            img = np.transpose(img,(1,2,0))
            img_HR = np.zeros((32,32,1,sampleNum))
            img_LR = np.zeros((32,32,1,sampleNum))
            for i in range(sampleNum):
                img_HR[:,:,0,i]=misc.imresize(img[:,:,i],(32,32))
                img_LR[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/scale),(32,32))
                
        if dataset == 'training':
            self.trainset = {'HR':img_HR,'LR':img_LR}
        else:
            self.testset =  {'HR':img_HR,'LR':img_LR}
            
    def saveToMatlab(self,dataset = "training"):                
    #        save samples on matlab form               
        import scipy.io
        if dataset == 'training':
            img_HR = self.trainset['HR']
            img_LR = self.trainset['LR']
        else:
            img_HR = self.testset['HR']
            img_LR = self.testset['LR']
            
        scipy.io.savemat('./data/'+dataset+'/MNIST_HR.mat', mdict={'MNIST_HR': img_HR})
        scipy.io.savemat('./data/'+dataset+'/MNIST_LR.mat', mdict={'MNIST_LR': img_LR})           
     
    def loadData(self,dataset = "training"): 
        if dataset == 'training':
            img_HR = self.trainset['HR']
            img_LR = self.trainset['LR']
        else:
            img_HR = self.testset['HR']
            img_LR = self.testset['LR']      
            
        img_HR = np.transpose(img_HR,(3,2,0,1))
        img_LR = np.transpose(img_LR,(3,2,0,1))
        img = {'HR':img_HR,'LR':img_LR}
        print('shape is (sampleNum,1,32,32)')
        return img
        
    def showMNIST(image):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1,1,1)
        imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()
    
    