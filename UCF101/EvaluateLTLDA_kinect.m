    clear;
    clc;
    
    % TODO: change the split_count here
    split_count = 1;
   
    CVAL = 1;
    
    charnum = 101;
    dim = 2048;
    classnum = charnum;
    rankdim = dim;

    mkdir('./datamat/results/');
    mkdir('./datamat/middata/');
   
    
    save_path = ['./datamat/TrTeSplit0' num2str(split_count) '_fortrain_kinect.mat'];
    load(save_path);
    
    max_iteration_num_ini = 30;
    max_iteration_num = max_iteration_num_ini;
    template_length = 8;
    band_factor = 2;

    
     %% construct sigmat
    %independent_train_label = [];
    independent_train_data = [];
    for j = 1:charnum
        for m = 1:trainsetnum(j)
            %independent_train_label = [independent_train_label; zeros(size(trainset{j}{m},1),1)+j];
            independent_train_data = [independent_train_data; trainset{j}{m}];
            %independent_class_train_data{j} = [independent_class_train_data{j}; trainset{j}{m}];
        end
    end    
    traindatamean = mean(independent_train_data, 1); 

    sigmat = zeros(dim,dim);
    total_num = 0;
    for c = 1:classnum
        for cons_sample_count = 1:trainsetnum(c)
            for j = 1:size(trainset{c}{cons_sample_count},1)
                temp_vector = trainset{c}{cons_sample_count}(j,:)-traindatamean;
                total_num = total_num + 1;
                sigmat = sigmat + temp_vector'*temp_vector;
            end
        end
    end
    sigmat = sigmat/total_num;

    [template_ini,alignpath_ini,ini_sigmaw] = TemplateClustering(classnum,trainset,trainsetnum,max_iteration_num_ini,template_length,band_factor);
    save(['./datamat/middata/inimidact_split' num2str(split_count) '_kinect.mat'],'template_ini','alignpath_ini','ini_sigmaw','sigmat');
    
    rM = getTransChange(ini_sigmaw,sigmat,rankdim);

    
    %% ½»²æÑéÖ¤

     i = 1;   
        count_temp = 0;
        for downdim = [500]
            count_temp = count_temp + 1;
            transMatrix_ini = rM(:,1:downdim);
            [transMatrix,template,alignpath,sigmaw] = getICMLTrans_change(sigmat,transMatrix_ini,trainset,trainsetnum,classnum,downdim,max_iteration_num,template_length,band_factor);
            save(['./datamat/middata/mid_' num2str(split_count) '_down_' num2str(downdim) '_kinect.mat'],'template','alignpath','sigmaw','transMatrix');
        end
