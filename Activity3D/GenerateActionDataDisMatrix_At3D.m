function  [xtraindis,yapp,xtestdis,ytest] = GenerateActionDataDisMatrix_At3D(all_SPDtype,opt)%
nSPDtype = length(all_SPDtype);
alldata = [];
alllabel = [];

for iSPDtype = 1:nSPDtype
    opt.SPDtype = all_SPDtype{iSPDtype};
    if(strcmp(opt.SPDtype,'RBF'))
        nrbf_sigma = length(opt.all_rbf_sigma);
        for irbf_sigma = 1:nrbf_sigma
            opt.rbf_sigma = opt.all_rbf_sigma(irbf_sigma);
            
            [Dis, label] = Load_SPD_Action_At3D(opt);
            
            alldata = [alldata,Dis];
            alllabel = [alllabel,label];
        end
    elseif(strcmp(opt.SPDtype,'POL'))
        npol_order = length(opt.all_pol_order);
        for ipol_order = 1:npol_order
            opt.pol_order = opt.all_pol_order(ipol_order);
            
            [Dis, label] = Load_SPD_Action_At3D(opt);
            
            alldata = [alldata,Dis];
            alllabel = [alllabel,label];
        end
    else
        [Dis, label] = Load_SPD_Action_At3D(opt);
        alldata = [alldata,Dis];
        alllabel = [alllabel,label];
    end
end

labeldiff = alllabel(:,2:end) - repmat(alllabel(:,1),1,size(alllabel(:,2:end),2));
labeldiff = abs(labeldiff);
if(sum(labeldiff(:))~=0)
    error('label_mismatch');
else
    alllabel = alllabel(:,1);
end

load(opt.dataset,'trainindex','testindex');
ndis = size(alldata,2);
for idis = 1:ndis
    xtraindis{1,idis} = alldata{idis}(trainindex,trainindex);
    xtestdis{1,idis} = alldata{idis}(testindex,trainindex);
end
yapp = alllabel(trainindex);
ytest = alllabel(testindex);

[xtraindis,yapp,xtestdis,ytest] = randdata(xtraindis,yapp,xtestdis,ytest);
end



function [xtraindis,yapp,xtestdis,ytest] = randdata(xtraindis,yapp,xtestdis,ytest)
ntrain = length(yapp);
ntest = length(ytest);
newtrainorder = randperm(ntrain);
newtestorder = randperm(ntest);

ndis = size(xtraindis,2);
for idis = 1:ndis
    xtraindis{1,idis} = xtraindis{idis}(newtrainorder,newtrainorder);
    xtestdis{1,idis} = xtestdis{idis}(newtestorder,newtrainorder);
end
yapp = yapp(newtrainorder);
ytest = ytest(newtestorder);

end

function [Dis, label,data] = Load_SPD_Action_At3D(opt)
load(opt.dataset,'trainindex','label','data');

downdim = opt.datadim;
load(['./datamat/middata/middown_' num2str(downdim) '_pro_ar.mat']);

ndata = length(data);
for i = 1:ndata
    [~,dim] = size(data{i});
            data{i} = data{i} - repmat(mean(data{i},2),1,dim);
            data{i} = l2norm(data{i});
    
    data{i} = transMatrix'*data{i};
end

% load('./datamat/trainset.mat');
% load('./datamat/trainsetnum.mat');
% load('./datamat/testset.mat');
% load('./datamat/testsetnum.mat');
% load('./datamat/testsetdata.mat');
% load('./datamat/testsetlabel.mat');
% load('./datamat/testsetdatanum.mat');
% load('./datamat/WSDAtrans_bfm.mat','W');

[data] = Feature_2_SPD(data,opt);
data = check_data(data);
[Dis]=mkldis_multiSPD(data,opt);

end

function [Dis, label,data] = Load_SPD_Action(opt)
load(opt.dataset,'trainindex','label','data')
[data] = Feature_2_SPD(data,opt);
data = check_data(data);
[Dis]=mkldis_multiSPD(data,opt);
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



function [fea,info] = Feature_2_SPD(data,opt)
if(iscell(data))
    ndata = length(data);
    info = size(ndata,1);
    for idata = 1:ndata
        [fea{idata,1},info(idata)] = SPDFeature(data{idata},opt);
    end
else
    [fea{1},info] = SPDFeature(data,opt);
end
end

function [fea,nfeature] = SPDFeature(data,opt)
if(opt.datadim ~= size(data,2))
    data = data';
end
if(isfield(opt,'ratio'))
    data = data(1:opt.ratio:end,:);
end

nfeature = size(data,1);
switch upper(opt.SPDtype)
    case 'KERCOV'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = kercov(data',opt);
    case 'COV'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = cov(data);
    case 'RBF'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = rbf_new_kernel(data',data',opt.rbf_sigma);
    case 'POL'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = poly_kernel(data',data',opt.pol_order);
    case 'MLP'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = mlp_kernel(data',data');
    case 'BHA'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = bhattacharyya_kernel(data',data');
    case 'JD'
        if(isfield(opt,'norm')&&opt.norm == 1)
            [~,dim] = size(data);
            data = data - repmat(mean(data,2),1,dim);
            data = l2norm(data);
        end
        fea = data;
end
end

function K = poly_kernel(u,v,polyOrder,varargin)
%POLY_KERNEL Polynomial kernel for SVM functions

% Copyright 2004-2012 The MathWorks, Inc.
% $Revision: 1.1.12.5 $  $Date: 2012/05/03 23:57:00 $

if nargin < 3 || isempty(polyOrder)
    polyOrder = 3; %default order
else
    if ~isscalar(polyOrder) || ~isnumeric(polyOrder)
        error(message('stats:poly_kernel:BadPolyOrder'));
    end
    if polyOrder ~= floor(polyOrder) || polyOrder < 1
        error(message('stats:poly_kernel:PolyOrderNotInt'))
    end
end

dotproduct = (u*v');

K = dotproduct;

for i = 2:polyOrder
    K = K.*(1 + dotproduct);
end
end

function kval = rbf_new_kernel(u,v,rbf_sigma,varargin)


if nargin < 3 || isempty(rbf_sigma)
    rbf_sigma = 1;
else
    if ~isscalar(rbf_sigma) || ~isnumeric(rbf_sigma)
        error(message('stats:rbf_kernel:RBFSigmaNotScalar'));
    end
    if rbf_sigma == 0
        error(message('stats:rbf_kernel:SigmaZero'));
    end
    
end

dissq = repmat(sqrt(sum(u.^2,2).^2),1,size(v,1))...
    -2*(u*v')+repmat(sqrt(sum(v.^2,2)'.^2),size(u,1),1);
dissq(dissq<0) = 0;


avesig = mean(mean(sqrt(dissq)));
rbf_sigma = rbf_sigma * avesig;
kval = exp(-(1/(2*rbf_sigma^2))*dissq);
end

function feature = l2norm(feature)
den = sum(feature.^2,1);
den(den == 0) = 1;
feature = feature ./repmat(sqrt(den),size(feature,1),1);
end



function [Dis]=mkldis_multiSPD(xapp,opt)
if(nargin<2)
    opt = [];
end
xsup=[];
[Dis] = svmkernel_SPD(xapp,xsup,opt);
end



function [Dis] = svmkernel_SPD(xapp,xsup,opt)
opt.usetheta = 0;
opt.thetascale = 1;

ndata = size(xapp, 2);
for idata = 1:ndata
    if(isempty(xsup))
        [~,~,Dis{idata}]  = LogEuclidean_kernel(xapp(:,idata),[],opt);
    else
        [~,~,Dis{idata}]  = LogEuclidean_kernel(xapp(:,idata),xsup(:,idata),opt);
    end
end

end



function [K,theta,loge] = LogEuclidean_kernel(data1,data2,opt)
if(size(data2,1)==0)
    if(isfield(opt,'eiglembda'))
        train_decomp = Decomposite_eig(data1,opt.eiglembda);
    else
        train_decomp = Decomposite_eig(data1);
    end
    [loge] = Log_E(train_decomp);
else
    if(isfield(opt,'eiglembda'))
        data_decomp1 = Decomposite_eig(data1,opt.eiglembda);
        data_decomp2 = Decomposite_eig(data2,opt.eiglembda);
    else
        data_decomp1 = Decomposite_eig(data1);
        data_decomp2 = Decomposite_eig(data2);
    end
    [loge] = Log_E(data_decomp1,data_decomp2);
end
if(isfield(opt,'usetheta')&&opt.usetheta == 0)
    
    theta = 1/mean(loge(:));
    if(isfield(opt,'thetascale')&&opt.thetascale ~= 0)
        theta = theta * opt.thetascale;
    end
    K = exp(-1*theta*loge);
else
    theta = opt.theta;
    K = exp(-1*theta*loge);
end
end

function [loge] = Log_E(A,B)
if(nargin<2)
    B = A;
    nA = size(A.V,1);
    nB = size(B.V,1);
    loge = zeros(nA,nB);
    for iA = 1:nA
        for iB = iA:nB
            tempA = A.V{iA}*diag(log(diag(A.D{iA})))*A.V{iA}';
            tempB = B.V{iB}*diag(log(diag(B.D{iB})))*B.V{iB}';
            if(any(size(tempA) - size(tempB)))
                disp('dimension mismatch');
            end
            dif = tempA - tempB;
            %dif = A.V{iA}*diag(log(diag(A.D{iA})))*A.V{iA}' - B.V{iB}*diag(log(diag(B.D{iB})))*B.V{iB}';
            loge(iA,iB) = sum(sum(dif.*dif));
            loge(iB,iA) = loge(iA,iB);
        end
    end
else
    nA = size(A.V,1);
    nB = size(B.V,1);
    loge = zeros(nA,nB);
    for iA = 1:nA
        for iB = 1:nB
            dif = A.V{iA}*diag(log(diag(A.D{iA})))*A.V{iA}' - B.V{iB}*diag(log(diag(B.D{iB})))*B.V{iB}';
            loge(iA,iB) = sum(sum(dif.*dif));
        end
    end
end


end


function [decomp] = Decomposite_eig(X,lembda)

% input:  X is a cell structure, each containing a PSD matrix;
%         index decides the set of eigenvalues to be included;
%
if(nargin<2)
    lembda = 1e-7;
end
n_X = length(X);
if(n_X == 0)
    disp('X is empty, error!');
    return;
end

% extract eigenvalues and check PSD for input;

V_x = cell(n_X,1);
D_x = cell(n_X,1);

for i = 1:n_X
    [V,D] = eig(X{i});
    V = real(V);
    D = real(D);
    if(any(diag(D)<lembda))
        D = abs(D);
        D = diag(diag(D) + lembda);
    end
    
    [Sorted_D,I] = sort(diag(D),'descend');
    V_x{i} = V(:,I);
    D_x{i} = abs(diag(Sorted_D));
end

decomp.V = V_x;
decomp.D = D_x;

end

