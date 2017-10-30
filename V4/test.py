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
parser.add_argument("--LR_layer", default="L0", type=str, help="the LR feature layer we are testing")
parser.add_argument("--HR_layer", default="L0", type=str, help="the LR feature layer we are testing")
parser.add_argument("--weight_thresholds", default=[0,0,0], type=list, help="The weight threshholds for each layer")
parser.add_argument("--weightFolder", default='MNIST_weight_0_0_0', type=str, help="The folder save the weight")
parser.add_argument("--LR_mode", default='LR_scale_4_interpo', type=str, help="downsample scale and interpolation or not")
parser.add_argument("--showIndex", default=2, type=int, help="index of the image you want to show")



def PSNR(noiseImg, noiseFreeImg):
    noiseImg=np.clip(noiseImg,0,255)
    imdff = noiseFreeImg - noiseImg
    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
    
    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
    return np.mean(20 * np.log10(255.0 / rmse))

opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = torch.load(opt.model)["model"]

import readMNIST
myDataset = readMNIST.DatasetMNIST()
myDataset.readMNIST(dataset = 'testing')
testset = myDataset.loadData(dataset = 'testing')
test_label = testset['HR'][:1000]
test_data = testset[opt.LR_mode][:1000]

psnr_bicubic =PSNR(test_data,test_label)

import layerExtractor
im_input=layerExtractor.forward(dataset=testset,mode=opt.LR_mode,layer=opt.LR_layer,\
                                savePatch=False,folder = opt.weightFolder,weight_thresholds = opt.weight_thresholds)
# im_input = torch.squeeze(im_input.data)

im_input = im_input/255.
im_input = im_input.float()[:1000]

# im_input = torch.unsqueeze(im_input,dim=1)
   
# im_input = Variable(im_input)

print(im_input.size())

if cuda:
    model = model.module.cuda()
    im_input = im_input.cuda()
 
start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out=out.cpu()
out = out*255.
test_pred=layerExtractor.inverse(forward_result=out,mode='HR',layer \
                              =opt.HR_layer,folder = opt.weightFolder,weight_thresholds = opt.weight_thresholds)

test_pred=test_pred.data.numpy()
print(test_pred.shape)
psnr_predicted = PSNR(test_pred,test_label)


print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))

fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(np.uint8(np.clip(test_label[opt.showIndex,0],0,255)),cmap=mpl.cm.Greys)
ax.set_title("HR")

ax = plt.subplot("132")
ax.imshow(np.uint8(np.clip(test_data[opt.showIndex,0],0,255)),cmap=mpl.cm.Greys)
ax.set_title("Input(bicubic)")

ax = plt.subplot("133")
ax.imshow(np.uint8(np.clip(test_pred[opt.showIndex,0],0,255)),cmap=mpl.cm.Greys)
ax.set_title("Output(vdsr)")
plt.show()

