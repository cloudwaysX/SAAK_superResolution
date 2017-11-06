#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:54:37 2017

@author: yifang
"""

import os
import numpy as np
from scipy import misc

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
#class DatasetMNIST():
#    
#    def __init__(self):
#        #shape is (size,size,1 sample number)
#        self.trainset = {'HR':None,'LR_scale_4_interpo':None,'LR_scale_6_interpo':\
#                         None,'LR_scale_4':None,'LR_scale_6':None} 
#        self.testset = {'HR':None,'LR_scale_4_interpo':None,'LR_scale_6_interpo':\
#                         None,'LR_scale_4':None,'LR_scale_6':None} 
#        
#    def readMNIST(self,dataset = "training", path = "."):
#        """
#        Python function for importing the MNIST data set.  It returns an iterator
#        of 2-tuples with the first element being the label and the second element
#        being a numpy.uint8 2D array of pixel data for the given image.
#        """
#        from scipy import misc
#        if dataset is "training":
#            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
#            sampleNum = 60000
#        elif dataset is "testing":
#            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
#            sampleNum = 10000
#        else:
#            raise(ValueError, "dataset must be 'testing' or 'training'")
#    #
#        with open(fname_img, 'rb') as fimg:
#            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
#            img = np.fromfile(fimg, dtype=np.uint8).reshape(sampleNum, rows, cols)
#            img = np.transpose(img,(1,2,0))
#            img_HR = np.zeros((32,32,1,sampleNum))
#            img_LR_scale_4_interpo = np.zeros((32,32,1,sampleNum))
#            img_LR_scale_6_interpo = np.zeros((32,32,1,sampleNum))
##            img_LR_scale_4 = np.zeros((8,8,1,sampleNum))
##            img_LR_scale_6 = np.zeros((4,4,1,sampleNum))
#
#            for i in range(sampleNum):
#                img_HR[:,:,0,i]=misc.imresize(img[:,:,i],(32,32))
#                img_LR_scale_4_interpo[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/4,interp='bicubic'),(32,32),interp='bicubic')
#                img_LR_scale_6_interpo[:,:,0,i]=misc.imresize(misc.imresize(img[:,:,i],1/6,interp='bicubic'),(32,32),interp='bicubic')
#        img_HR = np.transpose(img_HR,(3,2,0,1))
#        img_LR_scale_4_interpo = np.transpose(img_LR_scale_4_interpo,(3,2,0,1))
#        img_LR_scale_6_interpo = np.transpose(img_LR_scale_6_interpo,(3,2,0,1))
##        img_LR_scale_4 = np.transpose(img_LR_scale_4,(3,2,0,1))
##        img_LR_scale_6 = np.transpose(img_LR_scale_6,(3,2,0,1))
#        
#        if dataset == 'training':
#            self.trainset = {'HR':img_HR,'LR_scale_4_interpo':img_LR_scale_4_interpo,'LR_scale_6_interpo':\
#                         img_LR_scale_6_interpo} 
#        else:
#            self.testset = {'HR':img_HR,'LR_scale_4_interpo':img_LR_scale_4_interpo,'LR_scale_6_interpo':\
#                         img_LR_scale_6_interpo} 
#     
#    def loadData(self,dataset = "training"): 
#        if dataset == 'training':
#            img= self.trainset
#        else:
#            img= self.testset
#        print('shape is (sampleNum,1,32,32)')
#        return img
#        
#    def showMNIST(image):
#        """
#        Render a given numpy.uint8 2D array of pixel data.
#        """
#        from matplotlib import pyplot
#        import matplotlib as mpl
#        fig = pyplot.figure()
#        ax = fig.add_subplot(1,1,1)
#        imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
#        imgplot.set_interpolation('nearest')
#        ax.xaxis.set_ticks_position('top')
#        ax.yaxis.set_ticks_position('left')
#        pyplot.show()
#        
#    def writeTest(self,imageNum=10):
#        self.readMNIST(dataset='testing')
#        for i in range(10):
#            misc.imsave('/home/yifang/SAAK_superResolution/V4/data/temp_test/HR_'+str(i)+'.bmp',self.testset['HR'][i,0,:,:])
#            misc.imsave('/home/yifang/SAAK_superResolution/V4/data/temp_test/HR_'+str(i)+'_LR_scale_4_interpo.bmp',self.testset['LR_scale_4_interpo'][i,0,:,:])
#            misc.imsave('/home/yifang/SAAK_superResolution/V4/data/temp_test/HR_'+str(i)+'_LR_scale_6_interpo.bmp',self.testset['LR_scale_6_interpo'][i,0,:,:])       
  

def imageCropHelper(img,sizeL=256):
    L=int(sizeL/2-1)
    half_the_width = int(img.shape[0] / 2)
    half_the_height = int(img.shape[1] / 2)
    top = half_the_height - L; bot = half_the_height+L
    left = half_the_width -L; right = half_the_width+L
    return img[left:right+1,top:bot+1,:]
  
class DatasetLSUN():

    def __init__(self):
        #shape is (size,size,1 sample number)
        self.trainset = {'RGB':None,'HR':None,'LR_scale_4_interpo':None,\
                         'LR_scale_2_interpo':None,'LR_scale_3_interpo':None} 
        self.testset = {'RGB':None,'HR':None,'LR_scale_4_interpo':None,\
                         'LR_scale_2_interpo':None,'LR_scale_3_interpo':None} 
        
        
    def readLSUN(self,dataset = "training", path = ".",sampleNum=60000,cropSize = 256,inputSize = 64):
        print('shape for this {} is ({},1,{},{})'.format(dataset,sampleNum,inputSize,inputSize))
        import pickle 
        file = './lsun/LSUN_raw_data/'+dataset+'_'+str(sampleNum)+'_'+str(cropSize)+'_'+str(inputSize)+'.pickle'
        if os.path.exists(file):
            print('loading from existing dataset')
            if dataset=='training':
                with open(file,'rb') as f:
                    self.trainset = pickle.load(f)
            else:
                with open(file,'rb') as f:
                    self.testset = pickle.load(f)
            return
        if not os.path.exists('./lsun/LSUN_raw_data'):
            os.makedirs('./lsun/LSUN_raw_data')
        print('creating new dataset')
        import lmdb; import cv2
        if dataset == 'training':
            db_path = './lsun/bedroom_train_lmdb'
            sampleNum = sampleNum
        else:
            db_path = './lsun/bedroom_val_lmdb'
            sampleNum = sampleNum
        env = lmdb.open(db_path, map_size=1099511627776,max_readers=100, readonly=True)
        with env.begin(write=False) as txn:
            imgs_RGB = np.zeros((sampleNum,3,inputSize,inputSize))
            imgs_HR = np.zeros((sampleNum,1,inputSize,inputSize))
            imgs_LR_scale_2_interpo = np.zeros((sampleNum,1,inputSize,inputSize))
            imgs_LR_scale_3_interpo = np.zeros((sampleNum,1,inputSize,inputSize))
            imgs_LR_scale_4_interpo = np.zeros((sampleNum,1,inputSize,inputSize))
            cursor = txn.cursor()
            count=0
            for key, val in cursor:
                img_BGR = cv2.imdecode(np.fromstring(val, dtype=np.uint8), 1)
                img_BGR = imageCropHelper(img_BGR,sizeL=cropSize)
                img_Y = np.float64(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCR_CB)[:,:,0])
                img_RGB = np.float64(img_BGR[:,:,::-1])
                imgs_RGB[count,:,:,:]=np.transpose(misc.imresize(img_RGB,(inputSize,inputSize),interp = 'bicubic'),(2,0,1))  
#                imgs_RGB[count,:,:,:]=np.transpose(misc.imresize(img_RGB,(256,256),interp = 'bicubic'),(2,0,1))    
                img_Y=misc.imresize(img_Y,(inputSize,inputSize),interp = 'bicubic')
                imgs_HR[count,0,:,:] = img_Y
                imgs_LR_scale_2_interpo[count,0,:,:] = misc.imresize(misc.imresize(img_Y,1/2,interp = 'bicubic'),(inputSize,inputSize),interp='bicubic')
                imgs_LR_scale_3_interpo[count,0,:,:] = misc.imresize(misc.imresize(img_Y,1/3,interp = 'bicubic'),(inputSize,inputSize),interp='bicubic')
                imgs_LR_scale_4_interpo[count,0,:,:] = misc.imresize(misc.imresize(img_Y,1/4,interp = 'bicubic'),(inputSize,inputSize),interp='bicubic')
                count+=1
                if count==sampleNum:
                    break
        if dataset == 'training':
            self.trainset = {'RGB':imgs_RGB,'HR':imgs_HR,'LR_scale_4_interpo':imgs_LR_scale_4_interpo,'LR_scale_3_interpo':\
                         imgs_LR_scale_3_interpo,'LR_scale_2_interpo':imgs_LR_scale_2_interpo} 
            with open(file,'wb') as f:
                pickle.dump(self.trainset,f,pickle.HIGHEST_PROTOCOL)
        else:
            self.testset = {'RGB':imgs_RGB,'HR':imgs_HR,'LR_scale_4_interpo':imgs_LR_scale_4_interpo,'LR_scale_3_interpo':\
                         imgs_LR_scale_3_interpo,'LR_scale_2_interpo':imgs_LR_scale_2_interpo} 
            with open(file,'wb') as f:
                pickle.dump(self.testset,f,pickle.HIGHEST_PROTOCOL)
            
    def loadData(self,dataset = "training"): 
        if dataset == 'training':
            img= self.trainset
        else:
            img= self.testset
        return img
        
    def showLSUN(self,imageIndex,dataset = 'training',subdataset = 'HR'):
        """
        Render a given numpy.uint8 2D array of pixel data.
        """
        from matplotlib import pyplot as plt
        from matplotlib.cm import Greys
        if dataset == 'training':
            image = self.trainset[subdataset][imageIndex,:]
        else:
            image = self.testset[subdataset][imageIndex,:]
        print(image.shape)
        image = np.squeeze(np.transpose(image,(1,2,0)))
        plt.imshow(np.uint8(image),cmap=Greys)
        plt.title(dataset+'_'+subdataset)
        plt.show()
        return image
    
#def test(I): 
#    lsun = DatasetLSUN()
#    lsun.readLSUN(dataset = 'testing',sampleNum=300,inputSize=64)
#    lsun.showLSUN(I,dataset='testing', subdataset='RGB')
#    lsun.showLSUN(I,dataset='testing', subdataset='HR')
#    lsun.showLSUN(I,dataset='testing', subdataset='LR_scale_2_interpo')
#    lsun.showLSUN(I,dataset='testing', subdataset='LR_scale_3_interpo')
#    lsun.showLSUN(I,dataset='testing', subdataset='LR_scale_4_interpo')
##test(0)
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 21:54:37 2017

@author: yifang
"""

from os import listdir
from os.path import join
from skimage.util.shape import view_as_windows


"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath,win_size,cropSize,stride):
    img = misc.imread(filepath,mode='YCbCr')
    img = imageCropHelper(img,sizeL=cropSize)
    y= img[:,:,0]   
    win_size=win_size
    y = view_as_windows(y, (win_size,win_size),step=(stride,stride)).reshape(-1,win_size,win_size )
    return y

class DatasetBSD():
    
    def __init__(self):
        #shape is (size,size,1 sample number)
        self.trainset = {'HR':None,'LR_scale_4_interpo':None,'LR_scale_3_interpo':\
                         None} 
        self.testset = {'HR':None,'LR_scale_4_interpo':None,'LR_scale_3_interpo':\
                         None} 
        
    def readBSD(self,dataset = "training", path = ".",inputSize = 64,cropSize=[256,128],stride=16):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """
        from scipy import misc
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
    #
        for i in range(imageNum):
            y=load_img(fname_img[i],cropSize[1],cropSize[0],stride)
            if i==0:
                img=y
            else:
                img=np.concatenate((img,y))
        sampleNum=len(img)
        print('shape for this {} is ({},1,{},{})'.format(dataset,sampleNum,inputSize,inputSize))
        np.random.shuffle(img)
        img = np.transpose(img,(1,2,0))
        img_HR = np.zeros((inputSize,inputSize,1,sampleNum))
        img_LR_scale_4_interpo = np.zeros((inputSize,inputSize,1,sampleNum))
        img_LR_scale_3_interpo = np.zeros((inputSize,inputSize,1,sampleNum))

        for i in range(sampleNum):
            img_HR[:,:,0,i]=misc.imresize(img[:,:,i],(inputSize,inputSize),interp = 'bicubic')
            img_LR_scale_4_interpo[:,:,0,i]=misc.imresize(misc.imresize(img_HR[:,:,0,i],1/4,interp='bicubic'),(inputSize,inputSize),interp='bicubic')
            img_LR_scale_3_interpo[:,:,0,i]=misc.imresize(misc.imresize(img_HR[:,:,0,i],1/3,interp='bicubic'),(inputSize,inputSize),interp='bicubic')
            
        img_HR = np.transpose(img_HR,(3,2,0,1))
        img_LR_scale_4_interpo = np.transpose(img_LR_scale_4_interpo,(3,2,0,1))
        img_LR_scale_3_interpo = np.transpose(img_LR_scale_3_interpo,(3,2,0,1))
        
        if dataset == 'training':
            self.trainset = {'HR':img_HR,'LR_scale_4_interpo':img_LR_scale_4_interpo,'LR_scale_3_interpo':\
                         img_LR_scale_3_interpo} 
        else:
            self.testset = {'HR':img_HR,'LR_scale_4_interpo':img_LR_scale_4_interpo,'LR_scale_3_interpo':\
                         img_LR_scale_3_interpo} 
     
    def loadData(self,dataset = "training"): 
        if dataset == 'training':
            img= self.trainset
        else:
            img= self.testset
        print('shape is (sampleNum,1,32,32)')
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
        print(image.shape)
        plt.imshow(np.uint8(image),cmap=Greys)
        plt.title(dataset+'_'+subdataset)
        plt.show()
        return image

def test(Is): 
    lsun = DatasetBSD()
    lsun.readBSD(dataset = 'testing',)
    for I in Is:
        lsun.showBSD(I,dataset='testing', subdataset='HR')
        lsun.showBSD(I,dataset='testing', subdataset='LR_scale_4_interpo')
        lsun.showBSD(I,dataset='testing', subdataset='LR_scale_3_interpo')
        
if __name__ == "__main__":
    test([0,1,2,3,4])