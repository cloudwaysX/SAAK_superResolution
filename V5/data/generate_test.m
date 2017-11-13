clear;close all;

folder = 'Set5/test';
size_input = 32;
size_label = 32;
stride = 32;


%% scale factors
scale = [4];
%% downsizing
downsizes = 1;

%% initialization

count = 0;
margain = 0;

%% generate data   
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];

for i = 1:length(filepaths)
    count = 0;
    data = zeros(size_input, size_input, 1, 1);
    label = zeros(size_label, size_label, 1, 1);
    savepath = [fullfile(folder,filepaths(i).name),'_scale_',num2str(scale),'_',num2str(size_input),'_',num2str(stride),'.hd5'];
    image = imread(fullfile(folder,filepaths(i).name));
    if size(image,3)==3     
        image = rgb2ycbcr(image);
        image = im2double(image(:, :, 1));
        HR = modcrop(image, scale);
        [hei,wid] = size(HR);
        LR = imresize(imresize(HR,1/scale,'bicubic'),[hei,wid],'bicubic');
        filepaths(i).name
        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1

                subim_input = LR(x : x+size_input-1, y : y+size_input-1);
                subim_label = HR(x : x+size_label-1, y : y+size_label-1);

                count=count+1;

                data(:, :, 1, count) = subim_input;
                label(:, :, 1, count) = subim_label;
            end
        end
    end
    %% writing to HDF5
    chunksz = count;
    created_flag = false;
    totalct = 0;

    for batchno = 1:floor(count/chunksz)
        batchno
        last_read=(batchno-1)*chunksz;
        batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
        batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
% 
    h5disp(savepath);
end

