import argparse, os
import calcPCA_weight
import layerExtractor
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch



parser = argparse.ArgumentParser(description="classifier + LLS")

parser.add_argument("action", choices=['testInverse','testOnTrain','testOnVal','testOnTest'])
########preprocessing parameter##########################
parser.add_argument("--input_size", default = 32, type = int, help = "input size of training patch")
parser.add_argument("--stride", default = 32, type = int, help = "stride when cropping patches from original image")
parser.add_argument("--down_scale", default = 4, type = int, help = "how much downscale you want to get for Low Resolution Image")
########Kmean parameter##########################
parser.add_argument("--classifier_layer", default = "L1", help = "after which layer you want to add the classifier")
parser.add_argument("--n_cluster", default = 20, type=int, help = "how many clusters you want; if not using kmean, should be 1")
parser.add_argument("--classifier_weight_thresh", default = 1e-4, type = float, help = "the weight threshold to selet the feature used for classification, -1 means don't care")
parser.add_argument("--max_iter",default = 300, type = int, help = "the maxium iteration we want for kmean")
########SAAK parameter##########################
parser.add_argument("--end_layer", default = "L2", help = "The SAAK convolution stopped at this layer")
parser.add_argument("--zoomFactor", default = 4, type = int, help = "how much zoom for each SAAK transform")
parser.add_argument("--mapping_weight_thresh_HR", nargs = '+', default = [0,5e-5], type = float, help = "the weight threshold to drop the HR feature vector")
parser.add_argument("--mapping_weight_thresh_LR", nargs = '+', default = [0,1e-6], type = float, help = "the weight threshold to drop the LR feature vector")
########least square mapping##########################
parser.add_argument("--lsMapping_wholeSample", action="store_true", help="Calculate mapping matrix on whole image instead of patch?")
########Others##########################
parser.add_argument("--printNet", action="store_true", help="print out net structure?")
parser.add_argument("--printPcaW", action="store_true", help="print out PCA weight?")
parser.add_argument("--showImgIndex", default = -1, help = "which image you want to show")
parser.add_argument("--testImg", default = "head_GT", help = "Image used for testing")
parser.add_argument("--preTrained", action="store_true", help="already have model?")


opt = parser.parse_args()

assert len(opt.mapping_weight_thresh_LR) ==len(opt.mapping_weight_thresh_HR), 'Two weight threshold should have same length'
assert len(opt.mapping_weight_thresh_LR) == int(opt.end_layer[1]), 'Number of weight threshold does not match the conv layer'
assert opt.input_size % opt.stride == 0, 'Stirde that cannot be divided by input size is not suggested. May result to border issues when crop image'
if opt.input_size == 32 and opt.zoomFactor == 4: assert len(opt.mapping_weight_thresh_LR)!=3, 'For input size 32, can not do three layer 4*4 filter because 32/4/4/4 < 1'
if opt.preTrained: assert opt.action!='testInverse' and opt.action!='testOnTrain','pretrained mode only works in test on valildation dataset and test on test dataset'

def KmeanHelper(input,n_classifierUsedCompnent,k_means = None):
    print('start calcualte kmeans')
    half=(input.size()[1]-1)/2;half = int(half)
    # only perserve important dimension for k-mean classification
    input = input[:,list(range(n_classifierUsedCompnent+1))+list(range(half,half+n_classifierUsedCompnent))]
    sampleNum= input.size()[0]
    featureNum = input.size()[1]*input.size()[2]*input.size()[3]
    # arranged
    input = np.reshape(np.transpose(input.data.numpy(),(0,2,3,1)),(sampleNum,featureNum))
    
    if k_means is None:
        # return kmean if it haven't been calculated
        k_means = KMeans(n_clusters = opt.n_cluster,max_iter=opt.max_iter,n_jobs=-1)
        print('start fitting kmeans to data...')
        k_means.fit(input)
        return k_means
    else:
        # return the predicetion label if the kmean already be calculated
        print('start predicting labels using alrady existed kmean')
        return k_means.predict(input)

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
    if len(imdff.shape) == 2:
        rmse = np.sqrt(np.mean(imdff ** 2))
    else:
        rmse = np.sqrt(np.mean(imdff ** 2,axis=(1,2,3),keepdims=True))
            
    rmse = (rmse==0)*0.00255 + (rmse!=0)*rmse # set 0 diff equals to 100
    return np.mean(20 * np.log10(255.0 / rmse))

def PSNR_combinePatch(noiseImg, noiseFreeImg):
    noiseImg=np.clip(noiseImg,0,255)
    imdff = noiseFreeImg - noiseImg
    if len(imdff.shape) == 2:
        rmse = np.sqrt(np.mean(imdff ** 2))
    else:
        rmse = np.sqrt(np.mean(imdff ** 2,axis=(0,1,2,3),keepdims=True))
            
    rmse = (rmse==0)*0.00255 + (rmse!=0)*rmse # set 0 diff equals to 100
    return 20 * np.log10(255.0 / rmse)

def checkCluster(clusterI,dataset):
    print('cluster: {}'.format(clusterI))
    imgBatch_LR = dataset['LR_scale_{}_interpo'.format(opt.down_scale)][k_means.labels_==clusterI]
    imgBatch_HR = dataset['HR'][k_means.labels_==clusterI]
    clusterSize = imgBatch_HR.shape[0]
    for i in range(clusterSize):
        showImage(imgBatch_LR,imgBatch_HR,index=i)
        if i == 6: break
    
def lsMapping(clusterI,LR,HR=None,alreadyCalcMat = False):
    #arrange to shape (patch number, features)
    patchNum = LR.size()[0]*LR.size()[2]*LR.size()[3]
    sampleNum = LR.size()[0]
    if opt.lsMapping_wholeSample:
        featureNum_LR = LR.size()[1]*LR.size()[2]*LR.size()[3]
        LR_arranged = np.reshape(np.transpose(LR.data.numpy(),(0,2,3,1)),(sampleNum,featureNum_LR))
    else:
        featureNum_LR = LR.size()[1]
        LR_arranged = np.reshape(np.transpose(LR.data.numpy(),(0,2,3,1)),(patchNum,featureNum_LR))
    
    
    if not alreadyCalcMat:
        assert HR is not None, "Need HR value to calculate the mapping matrix"
        # calculate lslq 
        if opt.lsMapping_wholeSample:
            featureNum_HR = HR.size()[1]*LR.size()[2]*LR.size()[3]
            HR_arranged = np.reshape(np.transpose(HR.data.numpy(),(0,2,3,1)),(sampleNum,featureNum_HR))
        else:
            featureNum_HR = HR.size()[1]
            HR_arranged = np.reshape(np.transpose(HR.data.numpy(),(0,2,3,1)),(patchNum,featureNum_HR))
        results = np.linalg.lstsq(LR_arranged,HR_arranged)
#        print('residual is: {}'.format(results[1]))
        LR2HR_mat[clusterI] = results[0]
    
    #do the predicetion
    pred = np.dot(LR_arranged,LR2HR_mat[clusterI])
    if opt.lsMapping_wholeSample:
         pred = np.reshape(pred,(sampleNum,LR.size()[2],LR.size()[3],int(pred.shape[1]/LR.size()[2]/LR.size()[3])))
    else:
        pred = np.reshape(pred,(sampleNum,LR.size()[2],LR.size()[3],pred.shape[1]))
    pred = np.transpose(pred,(0,3,1,2))
    
    return pred



def PickleHelper(action,LR2HR_mat=None,k_means=None, n_classifierUsedCompnent=None):
    import pickle
    folder = './model/'+ str(opt.input_size)+'_'+str(opt.stride) + '/scale_' + str(opt.down_scale)+'/zoom_'+str(opt.zoomFactor)+'endL'+str(opt.end_layer)
    fileName = '/claL'+str(opt.classifier_layer)+'_clusterNum'+str(opt.n_cluster)+'_claW'+str(opt.classifier_weight_thresh)+'_iter'+str(opt.max_iter)+'_mappingW'+str(opt.mapping_weight_thresh_LR)+str(opt.mapping_weight_thresh_HR)
    if action == 'dump':
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder+fileName+'.pickle','wb') as f:
            pickle.dump({'LR2HR_mat':LR2HR_mat,'k_means':k_means,'n_classifierUsedCompnent': n_classifierUsedCompnent},f,pickle.HIGHEST_PROTOCOL)
    else:
        assert os.path.exists(folder),'Cannot find the trained model'
        with open(folder+fileName+'.pickle','rb') as f:
            tmp = pickle.load(f)
            LR2HR_mat = tmp['LR2HR_mat']
            k_means = tmp['k_means']
            n_classifierUsedCompnent = tmp['n_classifierUsedCompnent']
        return LR2HR_mat,k_means, n_classifierUsedCompnent
        
LR2HR_mat = {}    
k_means = None    

import readDataset
myDataset = readDataset.DatasetBSD()

if not opt.preTrained:
    myDataset.readBSD_fromMatlab(inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
    trainset = myDataset.loadData(dataset = 'training')
    
    pred = np.zeros(trainset['HR'].shape)
    
    if opt.action == 'testInverse': pred_LR = np.zeros(trainset['LR_scale_{}_interpo'.format(opt.down_scale)].shape)
        
    # =============================================================================
    # Training Part (Including testInverse and testOnTrain)
    # =============================================================================
    
    ########before classification, do SAAK together
    params_beforeClassify_LR = {"mapping_weight_threshold":opt.mapping_weight_thresh_LR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':None}   
    extracted_feats_beforeClassify_LR, n_classifierUsedCompnent = calcPCA_weight.calcW(trainset['LR_scale_{}_interpo'.format(opt.down_scale)],params_beforeClassify_LR,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer],mode = 'LR_scale_{}_interpo'.format(opt.down_scale),printPCAratio = opt.printPcaW)
    params_beforeClassify_HR = {"mapping_weight_threshold":opt.mapping_weight_thresh_HR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':None}  
    extracted_feats_beforeClassify_HR, _ = calcPCA_weight.calcW(trainset['HR'],params_beforeClassify_HR,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer],mode = 'HR',printPCAratio = opt.printPcaW)
    
    k_means = KmeanHelper(extracted_feats_beforeClassify_LR,n_classifierUsedCompnent)
    
    #################after classification, do SAAK seperately
    for clusterI in range(opt.n_cluster):
    #        checkCluster(clusterI,trainset)
        cur_cluster_features_LR = extracted_feats_beforeClassify_LR.data.numpy()[k_means.labels_==clusterI]
        params_afterClassify_LR = {"mapping_weight_threshold":opt.mapping_weight_thresh_LR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
        cur_cluster_features_LR, _ = calcPCA_weight.calcW(cur_cluster_features_LR,params_afterClassify_LR,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer],mode = 'LR_scale_{}_interpo'.format(opt.down_scale),printPCAratio = opt.printPcaW)
        
        cur_cluster_features_HR = extracted_feats_beforeClassify_HR.data.numpy()[k_means.labels_==clusterI]
        params_afterClassify_HR = {"mapping_weight_threshold":opt.mapping_weight_thresh_HR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
        cur_cluster_features_HR,_ = calcPCA_weight.calcW(cur_cluster_features_HR,params_afterClassify_HR,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer],mode = 'HR',printPCAratio = opt.printPcaW)
    
        print("the {} cluster has final feature shape HR={}, LR={}".format(clusterI,cur_cluster_features_HR.size(),cur_cluster_features_LR.size()))
        
        if opt.action != 'testInverse' and opt.action != 'testOnTrain': 
            # if in test on Validation or Test data mode, simply calculate mapping matrix
            lsMapping(clusterI,cur_cluster_features_LR,cur_cluster_features_HR)
        else:
             # if in test Inverse or test on Train data mode, calculate inverse direcly
            if opt.action == 'testInverse': 

                cur_cluster_features_HR_pred = cur_cluster_features_HR
                
                # for testInverse, see LR inverse back also
                params_afterClassify_LR={"mapping_weight_threshold":opt.mapping_weight_thresh_LR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
                cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_features_LR,params_afterClassify_LR,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer], mode='LR_scale_{}_interpo'.format(opt.down_scale),printNet = opt.printNet) 
#                del cur_cluster_features_LR
                params_beforeClassify_LR = {"mapping_weight_threshold":opt.mapping_weight_thresh_LR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
                cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_inverseFea_pred,params_beforeClassify_LR ,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer],  mode='LR_scale_{}_interpo'.format(opt.down_scale),printNet = opt.printNet)
                pred_LR[k_means.labels_==clusterI]= cur_cluster_inverseFea_pred.data.numpy()
                print('lslq',PSNR(cur_cluster_inverseFea_pred.data.numpy(),trainset['HR'][k_means.labels_==clusterI]))
                
            elif opt.action == 'testOnTrain':       
                cur_cluster_features_HR_pred = lsMapping(clusterI,cur_cluster_features_LR,cur_cluster_features_HR)
                del cur_cluster_features_LR;del cur_cluster_features_HR;
                cur_cluster_features_HR_pred=Variable(torch.Tensor(cur_cluster_features_HR_pred))
        
            params_afterClassify_HR={"mapping_weight_threshold":opt.mapping_weight_thresh_HR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
            cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_features_HR_pred,params_afterClassify_HR,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer], mode='HR',printNet = opt.printNet);
            del cur_cluster_features_HR_pred
            params_beforeClassify_HR = {"mapping_weight_threshold":opt.mapping_weight_thresh_HR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
            cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_inverseFea_pred,params_beforeClassify_HR ,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer], mode='HR',printNet = opt.printNet)
            pred[k_means.labels_==clusterI]= cur_cluster_inverseFea_pred.data.numpy()
            print('lslq',PSNR(cur_cluster_inverseFea_pred.data.numpy(),trainset['HR'][k_means.labels_==clusterI]))

    
    ##save model
    PickleHelper('dump',LR2HR_mat=LR2HR_mat,k_means=k_means, n_classifierUsedCompnent= n_classifierUsedCompnent)
#
    if opt.action == 'testOnTrain':    
        print('lslq',PSNR(pred,trainset['HR']))
        print('bicubic',PSNR(trainset['LR_scale_{}_interpo'.format(opt.down_scale)],trainset['HR']))
        showImage(trainset['LR_scale_{}_interpo'.format(opt.down_scale)],trainset['HR'],pred,index = opt.showImgIndex)
    elif opt.action == 'testInverse':
        print('HR back',PSNR(pred,trainset['HR']))
        print('LR back',PSNR(pred_LR,trainset['HR']))
        
else:
    LR2HR_mat,k_means, n_classifierUsedCompnent = PickleHelper('load')
        

    
# =============================================================================
# Testing Part (test on validation data and )
# =============================================================================
if not(opt.action == 'testInverse' or opt.action == 'testOnTrain'):
    if opt.action == 'testOnVal':
        myDataset.readBSD_fromMatlab(dataset='testing',inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
        validationOrtestSet = myDataset.loadData(dataset = 'testing')
    elif opt.action == 'testOnTest':
        import readDataset
        myDataset = readDataset.DatasetSet5()
        myDataset.readSet5_fromMatlab_singleImg(opt.testImg,inputSize=opt.input_size,stride = opt.stride,scale=opt.down_scale)
        validationOrtestSet = myDataset.loadData()
        
    pred = np.zeros(validationOrtestSet['HR'].shape) # redefine the shape of prediction here
    
    params_beforeClassify_LR = {"mapping_weight_threshold":opt.mapping_weight_thresh_LR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':None}
    extracted_feats_beforeClassify_LR = layerExtractor.forward(validationOrtestSet['LR_scale_{}_interpo'.format(opt.down_scale)],params_beforeClassify_LR,isbeforeCalssify=True,mode='LR_scale_{}_interpo'.format(opt.down_scale),in_out_layers =['L0',opt.classifier_layer], savePatch=False,printNet = opt.printNet)
    
    test_label = KmeanHelper(extracted_feats_beforeClassify_LR,n_classifierUsedCompnent,k_means = k_means)
    for clusterI in range(opt.n_cluster):
        if np.sum(test_label==clusterI) == 0: 
            print('testing data does not fall in cluster {}'.format(clusterI))
            continue
        cur_cluster_features_LR = extracted_feats_beforeClassify_LR.data.numpy()[test_label==clusterI]
        params_afterClassify_LR = {"mapping_weight_threshold":opt.mapping_weight_thresh_LR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
        cur_cluster_features_LR = layerExtractor.forward(cur_cluster_features_LR,params_afterClassify_LR,isbeforeCalssify=False,mode='LR_scale_{}_interpo'.format(opt.down_scale),in_out_layers =[opt.classifier_layer,opt.end_layer], savePatch=False,printNet = opt.printNet)
        
        cur_cluster_features_HR_pred = lsMapping(clusterI,cur_cluster_features_LR,alreadyCalcMat = True)
        cur_cluster_features_HR_pred=Variable(torch.Tensor(cur_cluster_features_HR_pred))
        
        params_afterClassify_HR={"mapping_weight_threshold":opt.mapping_weight_thresh_HR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
        cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_features_HR_pred,params_afterClassify_HR,isbeforeCalssify=False,in_out_layers=[opt.classifier_layer,opt.end_layer], mode='HR',printNet = opt.printNet); del cur_cluster_features_HR_pred
        params_beforeClassify_HR = {"mapping_weight_threshold":opt.mapping_weight_thresh_HR,"classifier_weight_threshold":opt.classifier_weight_thresh,"zoom factor":opt.zoomFactor,'cluster index':clusterI}
        cur_cluster_inverseFea_pred=layerExtractor.inverse(cur_cluster_inverseFea_pred,params_beforeClassify_HR ,isbeforeCalssify=True,in_out_layers=['L0',opt.classifier_layer], mode='HR',printNet = opt.printNet)
        pred[test_label==clusterI]= cur_cluster_inverseFea_pred.data.numpy()

    print('lslq',PSNR_combinePatch(pred,validationOrtestSet['HR']))
    print('bicubic',PSNR_combinePatch(validationOrtestSet['LR_scale_{}_interpo'.format(opt.down_scale)],validationOrtestSet['HR']))
    if opt.showImgIndex>0:
        showImage(validationOrtestSet['LR_scale_{}_interpo'.format(opt.down_scale)],validationOrtestSet['HR'],pred,index = opt.showImgIndex)
    

    
    


    
    
        
    





