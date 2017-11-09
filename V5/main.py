import argparse, os
import calcPCA_weight
import layerExtractor
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import Greys
from torch.autograd import Variable
import torch



parser = argparse.ArgumentParser(description="classifier + LLS")
parser.add_argument("--testInverse", action="store_true", help="Only used to test inverse in lossless and lossy mode?")
parser.add_argument("--classifier_layer", default = "L1", help = "after which layer you want to add the classifier")
parser.add_argument("--end_layer", default = "L2", help = "The SAAK convolution stopped at this layer")
parser.add_argument("--n_cluster", default = 10, help = "how many clusters you want; if not using kmean, should be 1")
parser.add_argument("--zoomFactor", default = 4, type = int, help = "how much zoom for each SAAK transform")
parser.add_argument("--mapping_weight_thresh", nargs = '+', default = [0,1e-5], type = int, help = "the weight threshold to drop the feature vector")
parser.add_argument("--classifier_weight_thresh", nargs = '+', default = [0,-1], type = int, help = "the weight threshold to selet the feature used for classification, -1 means don't care")


opt = parser.parse_args()
assert len(opt.mapping_weight_thresh) == len(opt.classifier_weight_thresh), 'two weight threshold should have same length'
assert len(opt.mapping_weight_thresh) == int(opt.end_layer[1]), 'number of weight threshold does not match the conv layer'

import readDataset
myDataset = readDataset.DatasetBSD()
myDataset.readBSD(inputSize=32,stride = 32)
trainset = myDataset.loadData(dataset = 'training')

LR2HR_mat = {}

def calcKmean(input,n_classifierUsedCompnent):
    print('start calcualte kmeans')
    half=(input.size()[1]-1)/2;half = int(half)
    # only perserve important dimension for k-mean classification
    input = input[:,list(range(n_classifierUsedCompnent+1))+list(range(half,half+n_classifierUsedCompnent))]
    sampleNum= input.size()[0]
    featureNum = input.size()[1]*input.size()[2]*input.size()[3]
    # arranged
    input = np.reshape(np.transpose(input.data.numpy(),(0,2,3,1)),(sampleNum,featureNum))
    k_means = KMeans(n_clusters = opt.n_cluster,max_iter=300,n_jobs=-1)
    print('start fitting kmeans to data...')
    k_means.fit(input)
    return k_means

def showImage(imgBatch_LR,imgBatch_HR,imgBatch_pred=None,index=0):
#    fig = plt.figure()
    ax = plt.subplot("131")
    ax.imshow(np.uint8(np.clip(imgBatch_HR[index,0],0,255)))
    ax.set_title("HR")

    ax = plt.subplot("132")
    ax.imshow(np.uint8(np.clip(imgBatch_LR[index,0],0,255)))
    ax.set_title("Input(bicubic)")
    
    if imgBatch_pred is not None:
        ax = plt.subplot("133")
        ax.imshow(np.uint8(np.clip(imgBatch_pred[index,0],0,255)))
        ax.set_title("Output(lslq)")
    plt.show()
    
def PSNR(noiseImg, noiseFreeImg):
    noiseImg=np.clip(noiseImg,0,255)
    imdff = noiseFreeImg - noiseImg
    print(noiseImg[0])
    rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
    
    rmse = (rmse==0)*0.001 + (rmse!=0)*rmse # set 0 diff equals to 100
    return np.mean(20 * np.log10(255.0 / rmse))

def checkCluster(clusterI,dataset):
    print('cluster: {}'.format(clusterI))
    imgBatch_LR = dataset['LR_scale_4_interpo'][k_means.labels_==clusterI]
    imgBatch_HR = dataset['HR'][k_means.labels_==clusterI]
    clusterSize = imgBatch_HR.shape[0]
    for i in range(clusterSize):
        showImage(imgBatch_LR,imgBatch_HR,index=i)
        if i == 6: break
    
def lsMapping(clusterI,LR,HR):
    #arrange to shape (patch number, features)
    patchNum = LR.size()[0]*LR.size()[2]*LR.size()[3]
    featureNum_LR = LR.size()[1]
    featureNum_HR = HR.size()[1]
    LR_arranged = np.reshape(np.transpose(LR.data.numpy(),(0,2,3,1)),(patchNum,featureNum_LR))
    HR_arranged = np.reshape(np.transpose(HR.data.numpy(),(0,2,3,1)),(patchNum,featureNum_HR))
    
    # calculate lwst 
    results = np.linalg.lstsq(LR_arranged,HR_arranged)
    print('residual is: {}'.format(results[1]))
    LR2HR_mat[clusterI] = results[0]
    
    #do the predicetion
    pred = np.dot(LR_arranged,LR2HR_mat[clusterI])
    
    pred = np.reshape(pred,(HR.size()[0],HR.size()[2],HR.size()[3],featureNum_HR))
    pred = np.transpose(pred,(0,3,1,2))
    
    return pred
    

#showImage(trainset['LR_scale_4_interpo'],trainset['HR'],None,2)


pred = np.zeros(trainset['HR'].shape)
params_beforeClassify = {"mapping_weight_threshold":[0],"classifier_weight_threshold":[1e-3],"zoom factor":opt.zoomFactor,'cluster index':None}

extracted_feats_beforeClassify_LR, n_classifierUsedCompnent = calcPCA_weight.calcW(trainset['LR_scale_4_interpo'],params_beforeClassify,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer],mode = 'LR_scale_4_interpo',printPCAratio = opt.testInverse)
k_means = calcKmean(extracted_feats_beforeClassify_LR,n_classifierUsedCompnent)


extracted_feats_beforeClassify_HR, _ = calcPCA_weight.calcW(trainset['HR'],params_beforeClassify,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer],mode = 'HR',printPCAratio = opt.testInverse)

for clusterI in range(opt.n_cluster):
#        checkCluster(clusterI,trainset)
    cur_cluster_features_LR = extracted_feats_beforeClassify_LR.data.numpy()[k_means.labels_==clusterI]
    params_afterClassify = {"mapping_weight_threshold":[0,1e-5],"classifier_weight_threshold":[1e-3,0],"zoom factor":opt.zoomFactor,'cluster index':clusterI}
    cur_cluster_features_LR, _ = calcPCA_weight.calcW(cur_cluster_features_LR,params_afterClassify,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer],mode = 'LR_scale_4_interpo',printPCAratio = opt.testInverse)
    
    cur_cluster_features_HR = extracted_feats_beforeClassify_HR.data.numpy()[k_means.labels_==clusterI]
    cur_cluster_features_HR,_ = calcPCA_weight.calcW(cur_cluster_features_HR,params_afterClassify,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer],mode = 'HR',printPCAratio = opt.testInverse)

    print('the {} cluster has final feature shape {}'.format(clusterI,cur_cluster_features_HR.size()))
    if opt.testInverse:
        del cur_cluster_features_LR
        cur_cluster_features_HR_pred = cur_cluster_features_HR
    else:        
        cur_cluster_features_HR_pred = lsMapping(clusterI,cur_cluster_features_LR,cur_cluster_features_HR)
        del cur_cluster_features_LR
        cur_cluster_features_HR_pred=Variable(torch.Tensor(cur_cluster_features_HR_pred))
        
    params_afterClassify={"mapping_weight_threshold":[0,1e-5],"classifier_weight_threshold":[1e-3,0],"zoom factor":opt.zoomFactor,'cluster index':clusterI}
    cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_features_HR_pred,params_afterClassify,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer], mode='HR',printNet = False); del cur_cluster_features_HR_pred
    params_beforeClassify = {"mapping_weight_threshold":[0],"classifier_weight_threshold":[1e-3],"zoom factor":opt.zoomFactor,'cluster index':clusterI}
    cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_inverseFea_pred,params_beforeClassify ,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer], mode='HR',printNet = False)
    pred[k_means.labels_==clusterI]= cur_cluster_inverseFea_pred.data.numpy()
            
print('lslq',PSNR(pred,trainset['HR']))
print('bicubic',PSNR(trainset['LR_scale_4_interpo'],trainset['HR']))
showImage(trainset['LR_scale_4_interpo'],trainset['HR'],pred)
#    
    
        
    





