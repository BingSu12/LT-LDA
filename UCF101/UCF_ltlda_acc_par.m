function UCF_ltlda_acc_par()
%% TODO: add path and change the split_count
processedpath = '/data/Bing/ResNext/code/datamat/';
addpath('~/liblinear-2.21/matlab');
addpath('~/libsvm-3.23/matlab');

split_count = 1;

%% classification
save_path = [processedpath 'TrTeSplit0' num2str(split_count) '_kinect.mat'];
load(save_path);

dimori = 2048;
dim = 2048*2;

acc_results = zeros(1,2);
acc_results_l2 = zeros(1,2);
acc_results_ori = zeros(1,2);
acc_results_ori_l2 = zeros(1,2);

temp_count = 0;
for downdim = [500 800]
    temp_count = temp_count + 1;
    load(['./datamat/middata/mid_' num2str(split_count) '_down_' num2str(downdim) '_kinect.mat']);
    %dimori = downdim;
    %dim = downdim*2;
    
    traindataori = zeros(trainsetdatanum,dimori);
    traindata = zeros(trainsetdatanum,dim);
    for i = 1:trainsetdatanum
        templedata = trainsetdata{i};  %*transMatrix;
        traindataori(i,:) = mean(templedata);
        traindata(i,:) = genRepresentation(templedata,1)';
    end
    
    testdataori = zeros(testsetdatanum,dimori);
    testdata = zeros(testsetdatanum,dim);
    for i = 1:testsetdatanum
        templedata = testsetdata{i};  %*transMatrix;
        testdataori(i,:) = mean(templedata);
        testdata(i,:) = genRepresentation(templedata,1)';
    end
    %save(['datasplit_' num2str(splitcount) '.mat'],'traindata','trainlables','testdata','testlables');
    
    C = 100;
    model = svmtrain(trainsetdatalabel, traindataori, sprintf('-t 0 -c %1.6f -q ',C));
    [~, acc, scores] = svmpredict(testsetdatalabel', testdataori ,model);
    acc_results_ori(temp_count) = acc(1); 
    
    C = 100;
    model = svmtrain(trainsetdatalabel, traindata, sprintf('-t 0 -c %1.6f -q ',C));
    [~, acc, scores] = svmpredict(testsetdatalabel', testdata ,model);
    acc_results(temp_count) = acc(1); 
    
    traindata = normalizeL2(traindata);
    testdata = normalizeL2(testdata);
    traindataori = normalizeL2(traindataori);
    testdataori = normalizeL2(testdataori);
    
    
    C = 100;
    model = svmtrain(trainsetdatalabel, traindataori, sprintf('-t 0 -c %1.6f -q ',C));
    [~, acc, scores] = svmpredict(testsetdatalabel', testdataori ,model);
    acc_results_ori_l2(temp_count) = acc(1); 
    
    C = 100;
    model = svmtrain(trainsetdatalabel, traindata, sprintf('-t 0 -c %1.6f -q ',C));
    [~, acc, scores] = svmpredict(testsetdatalabel', testdata ,model);
    acc_results_l2(temp_count) = acc(1); 
                 
end
acc_results
acc_results_ori
acc_results_l2
acc_results_ori_l2
save(['./datamat/results/mid_' num2str(split_count) '_result_ori_kinect.mat'],'acc_results','acc_results_ori','acc_results_l2','acc_results_ori_l2');
end


function Data = getNonLinearity(Data)
    %Data = sign(Data).*sqrt(abs(Data));
    %Data = vl_homkermap(Data',2,'kchi2');
    %Data =  sqrt(abs(Data));	                	
    Data =  sqrt(Data);	      
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end
    
    if normD == 1
        Data = normalizeL1(Data);
    end
    % in case it is complex, takes only the real part.	
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %d -s 11 -q',C) );
    w = model.w';
end

function X = normalizeL2(X)
	for i = 1 : size(X,1)
		if norm(X(i,:)) ~= 0
			X(i,:) = X(i,:) ./ norm(X(i,:));
		end
    end	   
end

function W = genRepresentation(data,CVAL)
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2); 			
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2); 			              
    W = [W_fow ; W_rev];
end
