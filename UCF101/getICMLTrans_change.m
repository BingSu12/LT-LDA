function [transMatrix,template,alignpath,sigmaw] = getICMLTrans_change(sigmat,transMatrix_ini,trainset,trainsetnum,classnum,downdim,max_iteration_num,template_length,band_factor)
    max_iteration_num_ini = max_iteration_num;
    traindownset = cell(1,classnum);
    for c = 1:classnum
        traindownset{c} = cell(1,trainsetnum(c));
    end
    dim = size(trainset{1}{1},2);
    transMatrix_old = zeros(dim,downdim);
    transMatrix = transMatrix_ini;
    %% iteration
    for ite = 1:max_iteration_num
        if sum(sum(abs(transMatrix-transMatrix_old))) < 10^(-6)
            disp('Outer loop:');
            ite
            break;
        end
        transMatrix_old = transMatrix;
        for c = 1:classnum
            for i = 1:trainsetnum(c)
                traindownset{c}{i} = trainset{c}{i}*transMatrix;
            end
        end       
        [template,alignpath,sigmaw] = TemplateClustering_subspace(classnum,trainset,traindownset,trainsetnum,max_iteration_num_ini,template_length,band_factor);
        transMatrix = getTransChange(sigmaw,sigmat,downdim);
    end
end