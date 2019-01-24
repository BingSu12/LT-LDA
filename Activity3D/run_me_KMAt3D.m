% This code implements kernel representation method for classification.
%
% Usage:
% 1. Download LIBSVM toobox by the following link:
% http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip
% 2. Run make.m under \libsvm-3.20\matlab file to compile LIBSVM
% 3. Add path \libsvm to matlab
% 4. Run run_me.m in matlab
% Note that the result obtained by this code is sligtly different from the
% result reported in our paper since a different SVM solver, SVM_KM, is used in our
% original experiment. LIBSVM is used here for conveniece.
% If users would like to use SVM_KM, just:
% 1 Download SVM_KM at http://asi.insa-rouen.fr/enseignants/~arakoto/toolbox/
% 2 Add path of SVM_KM by addpath(genpath('./SVM_KM'));
% 3 Replace the following line in our code
%    [kernel_accu] = Classification_kernels_libSVM(K_train,yapp,K_test,ytest,C);
%    with
%    [kernel_accu] = Classification_kernels(K_train,yapp,K_test,ytest,C); 
% Please note that the parameters in this code is used as an example. The
% parameters should be chosen by cross-validation to achieve optimal
% performance on each data set.

% Please cite the following paper if you use the code:
% Lei Wang, Jianjia Zhang, Luping Zhou, Chang Tang, and Wanqing Li. Beyond
% covariance: Feature representation with nonlinear kernel matrices. In
% ICCV, 2015
%
% For questions,  please conact:  Lei Wang leiw@uow.edu.au
% or the implementer Jianjia Zhang seuzjj@gmail.com
%
% The software is provided ''as is'' and without warranty of any kind,
% experess, implied or otherwise, including without limitation, any
% warranty of merchantability or fitness for a particular purpose.

clear;
%addpath(genpath('.\libsvm-3.20'));% the path of LIBSVM
addpath(genpath('E:\BING\ActionRecognition\FrameWideFeatures\SVM-KM'));
addpath(genpath('E:\BING\ActionRecognition\FrameWideFeatures\Kernel_representation_Code_for_release_v03'));

all_kernel_rep_type = {'COV' 'POL' 'RBF' 'KERCOV'};%
opt.type = 'exp';

filedir = {'MSR-DailyActivity3D'};
dims = [80];  %[10]  ; %[120];
ndataset = length(filedir);

idataset  = 1;
dataoption.datadim = dims(idataset);
dataoption.dataset = strcat(filedir{idataset},'.mat');
nkernel_rep_type = length(all_kernel_rep_type);
dataoption.norm = 1;

for ikernel_rep_type = 4  %1:nkernel_rep_type
    if(strcmp(all_kernel_rep_type(ikernel_rep_type),'RBF'))
        dataoption.all_rbf_sigma = 2.6;
        theta = 0.02;
        C = 10.^5;%([3:8]);
    elseif(strcmp(all_kernel_rep_type(ikernel_rep_type),'POL'))
%         theta = 0.1;
%         dataoption.all_pol_order = 4;
%         C = 10.^5;%([3:8]);
        theta = 0.005;
        dataoption.all_pol_order = 4;
        C = 10.^7;%([3:8]);
    elseif(strcmp(all_kernel_rep_type(ikernel_rep_type),'COV'))
        theta = 0.1;
        C = 10.^5;%([3:8]);
    elseif(strcmp(all_kernel_rep_type(ikernel_rep_type),'KERCOV'))
        %Activity3D
        %dataoption.sigma = 1.8; C= 10^7; theta = 0.005;
        %Action3D
        %dataoption.sigma = 1; C = 10^7; theta = 0.01;
        dataoption.sigma = 5; C = 10^7; theta = 0.005;
    end
    % load data, generate representations and calculate log_euclidean distance
    % matrices
    [xtraindis,yapp,xtestdis,ytest] = GenerateActionDataDisMatrix_At3D(all_kernel_rep_type(ikernel_rep_type),dataoption);%
    
    kerneloptionset={theta;1};
    
    % generate  kernels based on the distance matrices and
    [K_train,InfoKernel]=mklkernel_multiSPD_fromdis(xtraindis,kerneloptionset);
    [K_test]= mklkernel_multiSPD_fromdis(xtestdis,kerneloptionset,InfoKernel);
    %C = 10.^5;%([3:8]);
    % do classification
    %[kernel_accu] = Classification_kernels_libSVM(K_train,yapp,K_test,ytest,C);
    [kernel_accu] = Classification_kernels(K_train,yapp,K_test,ytest,C);
    fprintf('Classification Accuracy of %s is %.4f \n',all_kernel_rep_type{ikernel_rep_type},kernel_accu);

end





