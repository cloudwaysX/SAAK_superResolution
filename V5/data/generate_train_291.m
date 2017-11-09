clear;close all;

%folder = '../dataset/BSDS300/images/set5';
folder = 'test_sr'
savepath = '291_test_s8_64.h5';
size_input = 64;
size_label = 64;
stride = 64;
%%
% filepaths = [];
% filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
% for i = length(filepaths) - 50000:length(filepaths)
%      image = imread(fullfile(folder,filepaths(i).name));
%      [r,c,d] = size(image);
%      per = 0.2;
%      left = round(per*c);
%      right = c-left;
%      top = round(per*r);
%      bottom = r-top;
%      image = image(top:bottom, left:right, :);
%      image = imresize(image,[32 32],'bicubic');
%      image_s4 = imresize(image,[8 8],'bicubic');
%      %imwrite(image_s4,['./test/testsmall_',num2str(i),'_s4.jpg']);
%      image_s4 = imresize(image_s4,[32 32],'bicubic');
%      imwrite(image,['./testgt/test_',num2str(i),'_s4.jpg']);
%      imwrite(image_s4,['./test/testb_',num2str(i),'_s4.jpg']);
% end
%% scale factors
scale = 8;
%% downsizing
%downsizes = [1,0.7,0.5];
downsizes = 1;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
for i = 1:length(filepaths)
    for s = 1 : length(scale)
        for downsize = 1 : length(downsizes)
            image = imread(fullfile(folder,filepaths(i).name));
          
            if size(image,3)==3            
                image = rgb2ycbcr(image);
                        image = im2double(image(:, :, 1));

                        im_label = modcrop(image, scale(s));
                        [hei,wid] = size(im_label);
                        im_input = imresize(imresize(im_label,1/scale(s),'bicubic'),[hei,wid],'bicubic');
                        filepaths(i).name
                        for x = 1 : stride : hei-size_input+1
                            for y = 1 :stride : wid-size_input+1

                                subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
                                subim_label = im_label(x : x+size_label-1, y : y+size_label-1);
                                
                                count=count+1;

                                data(:, :, 1, count) = subim_input;
                                label(:, :, 1, count) = subim_label;
                            end
                        end
                
            end
            
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
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

h5disp(savepath);
