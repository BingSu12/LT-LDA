function [transMatrix,template,alignpath,sigmaw] = getICMLTrans(sigmat,transMatrix_ini,trainset,trainsetnum,classnum,downdim,max_iteration_num,template_length,band_factor)
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
%         switch modenum
%             case 1
%                 [M, lambda] = eig(sigmat - sigmaw);
%             case 2
%                 [M, lambda] = eig(sigmat - 2*sigmaw);
%             case 3
%                 [M, lambda] = eig(sigmat, sigmaw);
%             case 4
%                 [M, lambda] = eig(sigmat-sigmaw, sigmaw);
%         end
%         %[M, lambda] = eig(Sb, Sw);
%         [lambda, ind] = sort(diag(lambda), 'descend');
%         transMatrix = M(:,ind(1:min([downdim size(M, 2)])));
    end
end