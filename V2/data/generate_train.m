clear;close all;

savepath = 'train_L2.h5';
size_input = 4;
size_label = 4;

%% scale factors
scale = [4];
%% downsizing
downsizes = [1];


load('training/MNIST_LR_L2.mat');
load('training/MNIST_HR_L2.mat');
count = size(MNIST_LR_L2,4);
order = randperm(count);
data = MNIST_LR_L2(:,:,:,order)/255;
label = MNIST_HR_L2(:,:,:,order)/255;


%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
