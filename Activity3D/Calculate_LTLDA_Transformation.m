function Calculate_LTLDA_Transformation()

    %% 生成数据，提取特征
    clear;
    clc;
   
    CVAL = 1;
    
    charnum = 16;
    dim = 120;
    classnum = charnum;
    rankdim = dim;
    
    max_iteration_num_ini = 30;
    max_iteration_num = max_iteration_num_ini;
    template_length = 8;
    band_factor = 2;

    %load('./datamat/MSRAt3D_skel.mat');
    load('./datamat/MSRAt3D_skel_pro.mat');

     %% construct sigmat
    independent_train_data = [];
    for j = 1:charnum
        for m = 1:trainsetnum(j)
            independent_train_data = [independent_train_data; trainset{j}{m}];
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
    save(['./datamat/middata/inimidact_pro_ar.mat'],'template_ini','alignpath_ini','ini_sigmaw','sigmat');

    rM = getTransChange(ini_sigmaw,sigmat,rankdim);
    
    %% 交叉验证
    downdim = 80;
    transMatrix_ini = rM(:,1:downdim);
    [transMatrix,template,alignpath,sigmaw] = getICMLTrans_change(sigmat,transMatrix_ini,trainset,trainsetnum,classnum,downdim,max_iteration_num,template_length,band_factor);
%             transMatrix = getTransChange(sigmaw,sigmat,downdim);
    save(['./datamat/middata/middown_' num2str(downdim) '_pro_ar.mat'],'template','alignpath','sigmaw','transMatrix');
end

