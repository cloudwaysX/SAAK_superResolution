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

def imageCropHelper(img,inputSize):
    sz = img.shape
    sz = sz - np.mod(sz,inputSize) 
    return img[:sz[0],:sz[1]]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def img_Augment(img,win_size,stride,rot,scale):
    img=np.rot90(img,rot)
    y = imageCropHelper(img,win_size)
    y_LR=misc.imresize(misc.imresize(y,1/scale,interp='bicubic'),y.shape,interp='bicubic')
    y = view_as_windows(y, (win_size,win_size),step=(stride,stride)).reshape(-1,win_size,win_size )
    y_LR= view_as_windows(y_LR, (win_size,win_size),step=(stride,stride)).reshape(-1,win_size,win_size )
    return {'HR':y,'LR':y_LR}

class DatasetBSD():
    
    def __init__(self):
        #shape is (size,size,1 sample number)
        self.trainset = {} 
        self.testset = {} 
        
    def readBSD(self,dataset = "training", path = ".",inputSize = 64,stride=64,scale = 4):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """
        
        if dataset is "training":
            image_dir="BSDS300/images/train"
            fname_img=[join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
            imageNum = len(fname_img)
        elif dataset is "testing":
            image_dir="BSDS300/images/test"
            fname_img = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
            imageNum = len(fname_img)
        else:
            raise(ValueError, "dataset must be 'testing' or 'training'")
            
        img_HR=None
        img_LR=None
        for i in range(imageNum):
            img = misc.imread(fname_img[i],mode='YCbCr')[:,:,0]
            img=img[:,10:] # deal with border

            for rot in range(4):
                imgs = img_Augment(img,inputSize,stride,rot,scale)
                if img_HR is None:
                    img_HR=imgs['HR']
                    img_LR=imgs['LR']
                else:
                    img_HR=np.concatenate((img_HR,imgs['HR']))
                    img_LR=np.concatenate((img_LR,imgs['LR']))
        
        #change to shape (sampleNum,1,height,width) and to float        
        img_HR = np.expand_dims(np.float64(img_HR),axis=1)
        img_LR =  np.expand_dims(np.float64(img_LR),axis=1)
        
        #shuffle image
        import random
        reorder = list(range(img_HR.shape[0]))
        random.shuffle(reorder)
        img_HR = img_HR[reorder,:,:,:]
        img_LR = img_LR[reorder,:,:,:]
        
        print('shape for this {} is {})'.format(dataset,img_HR.shape))
                    
        if dataset == 'training':
            if not 'HR' in self.trainset: self.trainset['HR']=img_HR
            self.trainset['LR_scale_{}_interpo'.format(scale)]=img_LR
        else:
            if not 'HR' in self.testset: self.testset['HR']=img_HR
            self.testset['LR_scale_{}_interpo'.format(scale)]=img_LR
     
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

def test(Is): 
    lsun = DatasetBSD()
    lsun.readBSD(dataset = 'testing',inputSize=64,stride = 64)
    for I in Is:
        lsun.showBSD(I,dataset='testing', subdataset='HR')
        lsun.showBSD(I,dataset='testing', subdataset='LR_scale_4_interpo')
        
if __name__ == "__main__":
    test([0])