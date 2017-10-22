#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:25:31 2017

@author: yifang
"""

import argparse
import torch
from torch.autograd import Variable
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import misc

parser = argparse.ArgumentParser(description="PyTorch VDSR Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="HR_10", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--interpolation", action="store_true",  help="interpolate the LR?")
parser.add_argument("--mode", default="L", type=str, help="image type: YCbCr or black and white")
parser.add_argument("--LR_layer", default="L0", type=str, help="the LR feature layer we are testing")
parser.add_argument("--HR_layer", default="L2", type=str, help="the LR feature layer we are testing")

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    
def colorize(y, ycbcr): 
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img

opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

im_gt_ycbcr = imread("data/temp_test/" + opt.image + ".bmp", mode=opt.mode)
print(opt.interpolation)
if not opt.interpolation:
    im_b_ycbcr = imread("data/temp_test/"+ opt.image + "_scale_"+ str(opt.scale) + ".bmp", mode=opt.mode)
else:
    im_b_ycbcr = imread("data/temp_test/"+ opt.image + "_scale_"+ str(opt.scale) + "_interpo.bmp", mode=opt.mode)
    
if opt.mode == "YCbCr":
	im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
	im_b_y = im_b_ycbcr[:,:,0].astype(float) 
else:
	im_gt_y = im_gt_ycbcr.astype(float)
	im_b_y = im_b_ycbcr.astype(float) 	       

psnr_bicubic =PSNR(im_gt_y, misc.imresize(im_b_y,(32,32)),shave_border=opt.scale)

import LayerExtractor

testset = np.expand_dims(im_b_y,axis = 0); testset = np.expand_dims(testset,axis = 0);    
testset = {'LR':testset}
if not opt.interpolation:
    weight_folder = 'weight'
else:
    weight_folder = 'weight2'
im_input=LayerExtractor.forward(dataset=testset,datasetName="training"\
                                      ,mode='LR',layer=opt.LR_layer,savePatch=False,folder = 'weight2',interpolation=opt.interpolation)
im_input = torch.squeeze(im_input.data).numpy()

im_input = im_input/255.
im_input = torch.from_numpy(im_input).float()

while len(im_input.size()) < 4:
   im_input = torch.unsqueeze(im_input,dim=0)
   
im_input = Variable(im_input)

if cuda:
    model = model.module.cuda()
    im_input = im_input.cuda()
 
start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()

im_h_y = out.data[0].numpy().astype(np.float32)

im_h_y = im_h_y*255.
forward_result=torch.from_numpy(im_h_y)
forward_result = torch.unsqueeze(forward_result,dim=0);
im_h_y=LayerExtractor.inverse(forward_result=Variable(forward_result),mode='HR',layer =opt.HR_layer,folder = 'weight2')
im_h_y = im_h_y.data.numpy()[0]
im_h_y[im_h_y<0] = 0
im_h_y[im_h_y>255.] = 255.          

psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:],shave_border=opt.scale)

if opt.mode == "YCbCr":
	im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
	im_gt = Image.fromarray(im_gt_ycbcr, "YCbCr").convert("RGB")
	im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")
else:
	im_h = im_h_y
	im_gt = im_gt_ycbcr
	im_b = im_b_ycbcr

print("Scale=",opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt,cmap=mpl.cm.Greys)
ax.set_title("GT")

ax = plt.subplot("132")
ax.imshow(misc.imresize(im_b_y,(32,32)),cmap=mpl.cm.Greys)
ax.set_title("Input(bicubic)")

ax = plt.subplot("133")
print(im_h.shape)
ax.imshow(np.squeeze(im_h),cmap=mpl.cm.Greys)
ax.set_title("Output(vdsr)")
plt.show()
