#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:54:37 2017

@author: yifang
"""

import numpy as np
from scipy import misc

from os import listdir
from os.path import join
from skimage.util.shape import view_as_windows


"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


class DatasetBSD():
    
    def __init__(self):
        #shape is (size,size,1 sample number)
        self.trainset = {} 
        self.testset = {} 
        
            
    def readBSD_fromMatlab(self,dataset = "training", path = ".",inputSize = 9,stride=9,scale = 3):
        import os
        import h5py
        
        if dataset == "testing":
            file='test/scale_'+str(scale)+'_'+str(inputSize)+'_'+str(stride)+'.hd5'
        else:
            file='train/scale_'+str(scale)+'_'+str(inputSize)+'_'+str(stride)+'.hd5'
            
        name=os.path.join("data/BSD", file)
        assert os.path.exists(name), "{} not exist,need to generate patches using matlab first".format(name)
        hf=h5py.File(name)
        img_LR = np.float64(hf.get('data').value * 255.0)
        img_HR = np.float64(hf.get('label').value * 255.0)
        print('shape for this {} is {})'.format(dataset,img_HR.shape))
                    
        if dataset == 'training':
            if not 'HR' in self.trainset: self.trainset['HR']=img_HR
            self.trainset['LR_scale_{}_interpo'.format(scale)]=img_LR
            self.trainset['LR_scale_{}_interpo_diff'.format(scale)]=img_HR-img_LR
        else:
            if not 'HR' in self.testset: self.testset['HR']=img_HR
            self.testset['LR_scale_{}_interpo'.format(scale)]=img_LR
            self.testset['LR_scale_{}_interpo_diff'.format(scale)]=img_HR-img_LR
            


     
    def loadData(self,dataset = "training"): 
        if dataset == 'training':
            img= self.trainset
        else:
            img= self.testset
        return img
        
    def showBSD(self,imageIndex,dataset = 'training',subdataset = 'HR'):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        from matplotlib import pyplot as plt
        from matplotlib.cm import Greys
        if dataset == 'training':
            image = self.trainset[subdataset][imageIndex,:]
        else:
            image = self.testset[subdataset][imageIndex,:]
        image = np.squeeze(np.transpose(image,(1,2,0)))
        plt.imshow(np.uint8(image),cmap=Greys)
        plt.title(dataset+'_'+subdataset)
        plt.show()
        return image
    
class DatasetSet5():
    
    def __init__(self):
        #shape is (size,size,1 sample number)
        self.testset = {} 
            
    def readSet5_fromMatlab(self,path = ".",inputSize = 9,stride=9,scale = 3):
        import os
        import h5py
        file='test/scale_'+str(scale)+'_'+str(inputSize)+'_'+str(stride)+'.hd5'
            
        name=os.path.join("data/Set5", file)
        assert os.path.exists(name), "{} not exist,need to generate patches using matlab first".format(name)
        hf=h5py.File(name)
        img_LR = hf.get('data').value * 255.0
        img_HR = hf.get('label').value * 255.0
        print('shape for this tesing is {})'.format(img_HR.shape))
                    
        if not 'HR' in self.testset: self.testset['HR']=img_HR
        self.testset['LR_scale_{}_interpo'.format(scale)]=img_LR
            
    def readSet5_fromMatlab_singleImg(self,testImg, path = ".",inputSize = 64,stride=64,scale = 4):
        import os
        import h5py
        file='test/'+testImg+'.bmp_scale_'+str(scale)+'_'+str(inputSize)+'_'+str(stride)+'.hd5'
            
        name=os.path.join("data/Set5", file)
        assert os.path.exists(name), "{} not exist,need to generate patches using matlab first".format(name)
        hf=h5py.File(name)
        img_LR = hf.get('data').value * 255.0
        img_HR = hf.get('label').value * 255.0
        print('shape for this tesing is {})'.format(img_HR.shape))
                    
        if not 'HR' in self.testset: self.testset['HR']=img_HR
        self.testset['LR_scale_{}_interpo'.format(scale)]=img_LR

     
    def loadData(self,dataset = "training"): 

        img= self.testset
        return img
        
    def showSet5(self,imageIndex,dataset = 'training',subdataset = 'HR'):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        from matplotlib import pyplot as plt
        from matplotlib.cm import Greys
        if dataset == 'training':
            image = self.trainset[subdataset][imageIndex,:]
        else:
            image = self.testset[subdataset][imageIndex,:]
        image = np.squeeze(np.transpose(image,(1,2,0)))
        plt.imshow(np.uint8(image),cmap=Greys)
        plt.title(dataset+'_'+subdataset)
        plt.show()
        return image

def test(Is): 
    lsun = DatasetBSD()
    lsun.readBSD_fromMatlab(dataset = 'testing',inputSize=64,stride = 64)
    for I in Is:
        lsun.showBSD(I,dataset='testing', subdataset='HR')
        lsun.showBSD(I,dataset='testing', subdataset='LR_scale_4_interpo')
        
if __name__ == "__main__":
    test([0])