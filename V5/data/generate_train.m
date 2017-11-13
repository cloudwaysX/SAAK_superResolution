clear;close all;

folder = 'BSD';%change


size_input = 32;%change
size_label = 32;%change
stride = 32;%change

%% scale factors
scale = [4];%change

%% if augment
train = true; %change
if train
    folder = [folder,'/train'];
    flips = 3;
    degrees = 1;
else
    folder = [folder,'/test']
    flips = 1;
    degrees = 1;
end

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
if train
    savepath = [folder,'/scale_',num2str(scale),'_',num2str(size_input),'_',num2str(stride),'.hd5'];
else
    savepath = [folder,'/scale_',num2str(scale),'_',num2str(size_input),'_',num2str(stride),'.hd5'];
end

count = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];



for i = 1 : length(filepaths)
    for flip = 1: flips
        for degree = 1 : degrees
            for s = 1 : length(scale)
                image = imread(fullfile(folder,filepaths(i).name));

                if flip == 1
                    image = flipdim(image ,1);
                end
                if flip == 2
                    image = flipdim(image ,2);
                end

                image = imrotate(image, 90 * (degree - 1));


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
% 
h5disp(savepath);

%%

