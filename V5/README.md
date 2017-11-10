# SAAK_superResolution




commandline example:
train:
>> python main.py testOnVal --input_size 32 --stride 32 --down_scale 4 --classifier_layer L1 --n_cluster 10 --classifier_weight_thresh 1e-3 -1 --max_iter 300 --end_layer L2 --zoomFactor 4 --mapping_weight_thresh 0 1e-5 

pretrain:
python main.py testOnVal --input_size 32 --stride 32 --down_scale 4 --classifier_layer L1 --n_cluster 10 --classifier_weight_thresh 1e-3 -1 --max_iter 300 --end_layer L2 --zoomFactor 4 --mapping_weight_thresh 0 1e-5 --preTrained

Action Explain:
[testInverse]: this one direcly do the forward convolution on HR image (drop some feature if you are doing lossy) can then convert back. Use this to check if lossless can perfectly recover the image; And then use this to pre-select the mapping_weight_thresh, if the lossy invert back is already < 40 dB, then no need to further test on this parameter 

[testOnTrain] Test on trainset of BSD300.

[testOnVal] Test on validation dataset of BSD 300

[testOnTest] Test on Set5 by converting a 256*256 image to input_size*input_size small patches



