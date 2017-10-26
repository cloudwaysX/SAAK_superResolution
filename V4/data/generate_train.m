function generate_train(savepath,datasetName,HR_mode,LR_mode,layer,keepComp)   
    close all;

    if ~strcmp(layer,'L0')
        load([datasetName,'/',HR_mode,'_',layer,num2str(keepComp*100),'.mat']);
        load([datasetName,'/',LR_mode,'_',layer,num2str(keepComp*100),'.mat']);
    else
        load([datasetName,'/',HR_mode,'_',layer,'.mat']);
        load([datasetName,'/',LR_mode,'_',layer,'.mat']);
    end
    

    count = size(eval([HR_mode,'_',layer]),4);
    order = randperm(count);
    data = eval([LR_mode,'_',layer]);
    data = data(:,:,:,order)/255;
    label = eval([HR_mode,'_',layer]);
    label = label(:,:,:,order)/255;

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
end