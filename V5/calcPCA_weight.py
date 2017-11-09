from sklearn.decomposition import PCA
from skimage.util.shape import view_as_windows
import numpy as np

def Cal_W_PCA(X,mappingWeightThreshold,classifierWeightThreshold,zoomFactor,printPCAratio = False):
    if classifierWeightThreshold != -1:
        assert classifierWeightThreshold >= mappingWeightThreshold, 'can not use more then kept feature to do classification'

    reception_size = zoomFactor
    stride = (zoomFactor,zoomFactor,X.shape[1])


    assert len(X.shape)==4, "input images batch is not 4D"
    X=np.transpose(X,(2,3,0,1)) # width,height,sampleNum,depth
    assert X.shape[0] == X.shape[1], "input image is not square"
    input_width = X.shape[0]    
    input_depth = X.shape[3]
    sampleNum = X.shape[2]
    
    #######################################
    # extracted pathes from X, the shape of X_arranged is 
    #(input_width/stride,input_width/stride,1,reception_size,reception_size,input_depth)
    X = np.reshape(X,(input_width,input_width,input_depth*sampleNum)) #stack all samples alonge depth axis
    X_aranged = view_as_windows(X, window_shape=(reception_size, reception_size,input_depth),step=stride)

    # rearranged the extracted patches stacks
    # shape is (n_samples, n_features)
    patchNum = int(((input_width-reception_size)/stride[0]+1))**2*sampleNum
    featureNum = reception_size**2*input_depth
    X_aranged = np.reshape(X_aranged, (int(patchNum), int(featureNum)))
    
    X_aranged_whiten = X_aranged - np.mean(X_aranged,axis=1,keepdims=True)
    ############### AC ####################
    #apply PCA projection on the extracted patch samples
    #n_components == min(n_samples, n_features)
    pca = PCA(n_components=X_aranged_whiten.shape[1],svd_solver='full')
    pca.fit(X_aranged_whiten)
    #shape (n_components, n_features)
    W_pca_aranged = pca.components_
    if printPCAratio: 
        print(pca.explained_variance_ratio_)
    n_keptComponent = pca.explained_variance_ratio_ > mappingWeightThreshold
    W_pca_aranged = W_pca_aranged*((W_pca_aranged[:,[0]]>0)*2-1) #if the first item of W_pca_arranged is negative, make the whole vector posistive
    if np.sum(n_keptComponent)!=W_pca_aranged.shape[0]: #keep the DC weight
        W_pca_aranged = np.concatenate((W_pca_aranged[n_keptComponent],W_pca_aranged[[-1],:]))

    if classifierWeightThreshold is None:
        classfierUsedVecLength = None
    else:
        classfierUsedVecLength = np.sum(pca.explained_variance_ratio_ > classifierWeightThreshold)
    
    #shape as convolution weight required
    #shape: (output_depth,reception_size,reception_size,put_depth,)
    W_pca = np.reshape(W_pca_aranged,(W_pca_aranged.shape[0],reception_size,reception_size,input_depth))
    #shape: (output_depth,input_depth,reception_size,reception_size)
    W_pca = np.transpose(W_pca,(0,3,1,2))
    
    
    return {'W_pca':W_pca, 'n_keptComponent':W_pca_aranged.shape[0],'n_classifierUsedComponent': classfierUsedVecLength}

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

torch.cuda.manual_seed(1)

class Net(nn.Module):

    def __init__(self,zoomFactor):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.dc={'dc1':None,'dc2':None,'dc3':None}
        self.upsample = {'upsample1':None,'upsample2':None,'upsample3':None}
        self.conv = {'conv1':None,'conv2':None,'conv3':None}
        self.inChannel = {'in1':None,'in2':None,'in3':None}
        self.outChannel = {'out1':None,'out2':None,'out3':None}
        self.zoomFactor = zoomFactor
    
    def updateLayers(self,curL_in,curL_out,weight,layer):
        print('updating layer: '+layer)
        self.dc['dc'+layer]=nn.Conv2d(curL_in, 1, self.zoomFactor,stride=self.zoomFactor ) 
        self.upsample['upsample'+layer] = nn.Upsample(scale_factor=self.zoomFactor)
        self.conv['conv'+layer] = nn.Conv2d(curL_in, curL_out, self.zoomFactor,stride=self.zoomFactor ) 
        self.conv['conv'+layer].weight.data= torch.Tensor(weight)
        self.conv['conv'+layer].bias.data.fill_(0)
        self.dc['dc'+layer].weight.data = torch.Tensor(weight[[-1],:,:,:])
        self.dc['dc'+layer].bias.data.fill_(0)
        self.inChannel['in'+layer] = curL_in;
        self.outChannel['out'+layer] = curL_out;

    def forward(self,input,layer):
        print('forwarding at layer: '+layer)
        z_DC = self.dc['dc'+layer](input)
        n_feature = input.size()[1]*self.zoomFactor*self.zoomFactor
        z_mean = self.upsample['upsample'+layer](z_DC/np.sqrt(n_feature))
        z_AC = self.conv['conv'+layer](input-z_mean)
        z_AC = z_AC[:,:-1,:,:]
        A_1 = F.relu(z_AC);A_2=F.relu(-z_AC)
        A = torch.cat((z_DC,A_1,A_2),dim = 1)
        return A

def calcW(dataset,params,isbeforeCalssify,in_out_layers=['L0','L3'],mode = 'HR',printPCAratio = False):

    assert in_out_layers[0] <= in_out_layers[1],'in layer must before or the same as the out layer'
    mappingWeightThresholds = params["mapping_weight_threshold"]
    classifierWeightThresholds = params["classifier_weight_threshold"]
    zoomFactor = params['zoom factor']
    clusterI = params['cluster index']

    import os
    temp = in_out_layers[0]+'_2_'+in_out_layers[1]
    folder = './weight/'+'/zoom_'+str(zoomFactor)+'/'+mode+'/'+temp
    if not os.path.exists(folder):
        os.makedirs(folder)

    net = Net(zoomFactor=zoomFactor)


    X=torch.Tensor(dataset) #mode can be 'HR', 'LR_scale_X'
    X=Variable(X)

    def L1(input):
        curInCha = input.size()[1]
        result= Cal_W_PCA(input.data.numpy(),mappingWeightThresholds[0],classifierWeightThresholds[0],zoomFactor,printPCAratio = printPCAratio)
        W_pca=result['W_pca']
        curOutChar = result['n_keptComponent']
        n_classifierUsedComponent=result['n_classifierUsedComponent']
        if isbeforeCalssify:
            np.save(folder + '/L1_'+str(mappingWeightThresholds[0]) + '_beforeClassifier.npy',W_pca)
        else:
             np.save(folder + '/L1_'+str(mappingWeightThresholds[0]) + '_'+str(clusterI)+'.npy',W_pca)
        net.updateLayers(curInCha,curOutChar,W_pca,'1')
        A1 = net.forward(input,'1')

        return A1,n_classifierUsedComponent

    def L2(input):
        curInCha = input.size()[1]
        result= Cal_W_PCA(input.data.numpy(),mappingWeightThresholds[1],classifierWeightThresholds[1],zoomFactor,printPCAratio = printPCAratio)
        W_pca=result['W_pca']
        curOutChar = result['n_keptComponent']
        n_classifierUsedComponent=result['n_classifierUsedComponent']
        if isbeforeCalssify:
            np.save(folder + '/L2_'+str(mappingWeightThresholds[1]) + '_beforeClassifier.npy',W_pca)
        else:
             np.save(folder + '/L2_'+str(mappingWeightThresholds[1]) + '_'+str(clusterI)+'.npy',W_pca)
        net.updateLayers(curInCha,curOutChar,W_pca,'2')
        A2 = net.forward(input,'2')

        return A2,n_classifierUsedComponent

    def L3(input):
        curInCha = input.size()[1]
        result= Cal_W_PCA(input.data.numpy(),mappingWeightThresholds[2],classifierWeightThresholds[2],zoomFactor,printPCAratio = printPCAratio)
        W_pca=result['W_pca']
        curOutChar = result['n_keptComponent']
        n_classifierUsedComponent=result['n_classifierUsedComponent']
        if isbeforeCalssify:
            np.save(folder + '/L3_'+str(mappingWeightThresholds[2]) + '_beforeClassifier.npy',W_pca)
        else:
             np.save(folder + '/L3_'+str(mappingWeightThresholds[2]) + '_'+str(clusterI)+'.npy',W_pca)
        net.updateLayers(curInCha,curOutChar,W_pca,'3')
        A3 = net.forward(input,'3')

        
        return A3,n_classifierUsedComponent

    if in_out_layers[1]=='L1':
        result,n_classifierUsedComponent = L1(X)
    elif in_out_layers[1] == 'L2':
        if in_out_layers[0] == 'L0': 
            A1,_ = L1(X); del X
            result,n_classifierUsedComponent = L2(A1); 
        else:
            result,n_classifierUsedComponent = L2(X)
    else:
        if in_out_layers[0] == 'L0': 
            A1,_ = L1(X); del X
            A2,_ = L2(A1); del A1
            result,n_classifierUsedComponent = L3(A2)
        elif in_out_layers[0] == 'L1':
            A2,_ = L2(X); del X
            result,n_classifierUsedComponent = L3(A2)
        else:
            result,n_classifierUsedComponent = L3(X)

    np.save(folder+'/'+str(mappingWeightThresholds)+'struc.npy',(net.inChannel,net.outChannel))
    return result,n_classifierUsedComponent