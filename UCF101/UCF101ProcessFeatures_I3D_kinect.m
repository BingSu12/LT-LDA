%% TODO: set the feature path and the output path
featuresetpath = '/data/Bing/ResNext/video-classification-3d-cnn-pytorch-master/features/';
processedpath = '/data/Bing/ResNext/code/datamat/';

%%
classnum = 101;
dim = 2048;

class_id_temp = importdata('./ucfTrainTestlist/classInd.txt');
class_id = cell(size(class_id_temp,1),2);
for i = 1:size(class_id_temp,1)
    s_p = strfind(class_id_temp{i},' ');
    class_id{i,1} = str2num(class_id_temp{i}(1:s_p(1)-1));
    class_id{i,2} = class_id_temp{i}(s_p(1)+1:end);
end

for split_count = 1:3
    train_path_name = ['./ucfTrainTestlist/trainlist0' num2str(split_count) '.txt'];
    train_temp = importdata(train_path_name);
    trainsetdatanum = size(train_temp.data,1);
    trainsetdata = cell(1,trainsetdatanum);
    trainsetdatalabel = train_temp.data;
    trainsetnum = zeros(1,classnum);
    
    avelength = 0;
    for i = 1:trainsetdatanum
        trainsetnum(train_temp.data(i)) = trainsetnum(train_temp.data(i)) + 1;
        pos1 = strfind(train_temp.textdata{i},'/');
        pos2 = strfind(train_temp.textdata{i},'.');
        matpath = [featuresetpath train_temp.textdata{i}(pos1(1)+1:pos2(1)-1) '.mat'];
        load(matpath);
        frmnum = size(feature,2);
        trainsetdata{i} = zeros(frmnum,dim);
        for f = 1:frmnum
            trainsetdata{i}(f,:) = feature{f}.features;
        end
        avelength = avelength + frmnum;
    end
    avelength = avelength/trainsetdatanum;
    
    trainset = cell(1,classnum);
    for c = 1:classnum
        trainset{c} = cell(1,trainsetnum(c));
        local_temp = 0;
        for i = 1:trainsetdatanum
            if train_temp.data(i)==c
                local_temp = local_temp + 1;
                trainset{c}{local_temp} = trainsetdata{i};
            end
        end
    end
    
    test_path_name = ['./ucfTrainTestlist/testlist0' num2str(split_count) '.txt'];
    test_temp = importdata(test_path_name);
    testsetdatanum = size(test_temp,1);
    testsetdata = cell(1,testsetdatanum);
    testsetdatalabel = zeros(1,testsetdatanum);
    for i = 1:testsetdatanum
        pos1 = strfind(test_temp{i},'/');
        pos2 = strfind(test_temp{i},'.');
        matpath = [featuresetpath test_temp{i}(pos1(1)+1:pos2(1)-1) '.mat'];
        load(matpath);
        frmnum = size(feature,2);
        testsetdata{i} = zeros(frmnum,dim);
        for f = 1:frmnum
            testsetdata{i}(f,:) = feature{f}.features;
        end
        classname = test_temp{i}(1:pos1(1)-1);  
        for k = 1:classnum
            if strcmp(classname,class_id{k,2})
                testsetdatalabel(i) = class_id{k,1};
                break;
            end
        end
    end 
    
    save_path = [processedpath 'TrTeSplit0' num2str(split_count) '_fortrain_kinect.mat'];
    save(save_path,'trainset','trainsetnum','avelength','-v7.3');
    save_path = [processedpath 'TrTeSplit0' num2str(split_count) '_kinect.mat'];
    save(save_path,'trainsetdata','trainsetdatanum','trainsetdatalabel','testsetdata','testsetdatanum','testsetdatalabel','-v7.3');
end
