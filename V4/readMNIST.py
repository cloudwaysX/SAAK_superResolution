#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:54:37 2017

@author: yifang
"""

import os
import struct
import numpy as np
from scipy import misc

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
class DatasetMNIST():
    
    def __init__(self):
        #shape is (size,size,1 sample number)
        self.trainset = {'HR':None,'LR_scale_4_interpo':None,'LR_scale_6_interpo':\
                         None,'LR_scale_4':None,'LR_scale_6':None} 
        self.testset = {'HR':None,'LR_scale_4_interpo':None,'LR_scale_6_interpo':\
                         None,'LR_scale_4':None,'LR_scale_6':None} 
        
    def readMNIST(self,dataset = "training", path = "."):
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
            img_LR_scale_4_interpo = np.zeros((32,32,1,sampleNum))
            img_LR_scale_6_interpo = np.zeros((32,32,1,sampleNum))
#            img_LR_scale_4 = np.zeros((8,8,1,sampleNum))
#            img_LR_scale_6 = np.zeros((4,4,1,sampleNum))

            for i in range(sampleNum):
                img_HR[:,:,0,i]=misc.imresize(img[:,:,i],(32,32))
                img_LR_scale_4_interpo[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/4,interp='bicubic'),(32,32),interp='bicubic')
                img_LR_scale_6_interpo[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/6,interp='bicubic'),(32,32),interp='bicubic')
#                img_LR_scale_4[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/4,interp='bicubic'),(8,8),interp='bicubic')
#                img_LR_scale_8[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/6,interp='bicubic'),(4,4),interp='bicubic')
        img_HR = np.transpose(img_HR,(3,2,0,1))
        img_LR_scale_4_interpo = np.transpose(img_LR_scale_4_interpo,(3,2,0,1))
        img_LR_scale_6_interpo = np.transpose(img_LR_scale_6_interpo,(3,2,0,1))
#        img_LR_scale_4 = np.transpose(img_LR_scale_4,(3,2,0,1))
#        img_LR_scale_6 = np.transpose(img_LR_scale_6,(3,2,0,1))
        
        if dataset == 'training':
            self.trainset = {'HR':img_HR,'LR_scale_4_interpo':img_LR_scale_4_interpo,'LR_scale_6_interpo':\
                         img_LR_scale_6_interpo} 
        else:
            self.testset = {'HR':img_HR,'LR_scale_4_interpo':img_LR_scale_4_interpo,'LR_scale_6_interpo':\
                         img_LR_scale_6_interpo} 
     
    def loadData(self,dataset = "training"): 
        if dataset == 'training':
            img= self.trainset
        else:
            img= self.testset
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
        
    def writeTest(self,imageNum=10):
        self.readMNIST(dataset='testing')
        for i in range(10):
            misc.imsave('/home/yifang/SAAK_superResolution/V4/data/temp_test/HR_'+str(i)+'.bmp',self.testset['HR'][i,0,:,:])
            misc.imsave('/home/yifang/SAAK_superResolution/V4/data/temp_test/HR_'+str(i)+'_LR_scale_4_interpo.bmp',self.testset['LR_scale_4_interpo'][i,0,:,:])
            misc.imsave('/home/yifang/SAAK_superResolution/V4/data/temp_test/HR_'+str(i)+'_LR_scale_6_interpo.bmp',self.testset['LR_scale_6_interpo'][i,0,:,:])
        
    
    