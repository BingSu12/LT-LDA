function construct_state_of_art()
load('MSR-DailyActivity3D.mat','trainindex','testindex','label','data');
nfeature = size(data,1);
classnum = 16;
trainset = cell(1,classnum);
trainsetnum = zeros(1,classnum);
testsetdatanum = size(testindex,1);
testsetdata = cell(1,testsetdatanum);
trainsetdatanum = size(testindex,1);
testsetlabel = zeros(1,testsetdatanum);

oriset = 0;

if oriset == 1

for c = 1:classnum
    perclasscount = 0;
    for i = 1:trainsetdatanum
        k = trainindex(i);
        if label(k)==c
            perclasscount = perclasscount + 1
            trainset{c}{perclasscount} = data{k}';
        end
    end
    trainsetnum(c) = perclasscount;
end

testcount = 0;
for i = 1:testsetdatanum
    k=testindex(i);
    testcount = testcount + 1;
    testsetdata{testcount} = data{k}';
    testsetlabel(testcount) = label(k);
end

save('./datamat/MSRAt3D_skel.mat','trainset','trainsetnum','testsetdatanum','testsetdata');

else
    for c = 1:classnum
        perclasscount = 0;
        for i = 1:trainsetdatanum
            k = trainindex(i);
            if (label(k)==c)
                perclasscount = perclasscount + 1;
                trainset{c}{perclasscount} = processFea(data{k})';
            end
        end
        trainsetnum(c) = perclasscount;
    end

    testcount = 0;
    for i = 1:testsetdatanum
        k=testindex(i);
        testcount = testcount + 1;
        testsetdata{testcount} = processFea(data{k})';
    end
    
    save('./datamat/MSRAt3D_skel_pro.mat','trainset','trainsetnum','testsetdatanum','testsetdata');
end

% if(isfield(opt,'norm')&&opt.norm == 1)
%     [~,dim] = size(data);
%     data = data - repmat(mean(data,2),1,dim);
%     data = l2norm(data);
% end
% fea = kercov(data',opt);

end

function data = processFea(data)
[~,dim] = size(data);
    data = data - repmat(mean(data,2),1,dim);
    data = l2norm(data);
end
        
function feature = l2norm(feature)
den = sum(feature.^2,1);
den(den == 0) = 1;
feature = feature ./repmat(sqrt(den),size(feature,1),1);
end

function [data,ex_index] = check_data(data,lembda)
if(nargin<2)
    lembda = 1e-7;
end
ndata = size(data,1);
ex_index = [];
for idata = 1:ndata
    if(any(isnan(data{idata})))
        ex_index = [ex_index;idata];
        continue;
    end
    [v,d] = eig(data{idata});
    %v = real(v);
    %d = real(d);
    if(any(diag(d)<lembda))
        %d = abs(d);
        d = diag(diag(d) + lembda);
    end
    data{idata} = v*d*v';
end
end        