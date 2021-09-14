clear all;
close all;

%% load data


train_images = [];
train_labels = [];
root_dir = 'D:/data_object_image_2';
data_set = 'training';
cam = 2; % 2 = left color camera
pedresize_dir = fullfile(root_dir,[data_set '/pedresize_' num2str(cam)]);
other_dir = fullfile(root_dir,[data_set '/other_' num2str(cam)]);






size_train = 6000;
size_test = 1000;
width = 30;
height = 60;
channel = 3;
train_data = zeros(height,width,channel,size_train*4);
test_data = zeros(height,width,channel,size_test*4);
h1 = fspecial('gaussian',3,3);
h2 = fspecial('gaussian',7,3);
h3 = fspecial('gaussian',11,3);
for i = 1:size_train/2
    I = imread(sprintf('%s/%06d.png',pedresize_dir,i));
    train_data(:,:,:,i) = I;
    train_data(:,:,:,i+size_train) = imfilter(I,h1,'replicate');
    train_data(:,:,:,i+2*size_train) = imfilter(I,h2,'replicate');
    train_data(:,:,:,i+3*size_train) = imfilter(I,h3,'replicate');
end
for i = 1:size_train/2
    I = imread(sprintf('%s/%06d.png',other_dir,i));
    train_data(:,:,:,i+size_train/2) = I;
    train_data(:,:,:,i+size_train/2+size_train) = imfilter(I,h1,'replicate');
    train_data(:,:,:,i+size_train/2+2*size_train) = imfilter(I,h2,'replicate');
    train_data(:,:,:,i+size_train/2+3*size_train) = imfilter(I,h3,'replicate');
end


for i = 1:size_test/2
    I = imread(sprintf('%s/%06d.png',pedresize_dir,i+3000));
    test_data(:,:,:,i) = I;
    test_data(:,:,:,i + size_test) =  imfilter(I,h1,'replicate');
    test_data(:,:,:,i + 2*size_test) =  imfilter(I,h2,'replicate');
    test_data(:,:,:,i + 3*size_test) =  imfilter(I,h3,'replicate');
    
end
for i = 1:size_test/2
    I = imread(sprintf('%s/%06d.png',other_dir,i+3000));
    test_data(:,:,:,i+size_test/2) = I;
    test_data(:,:,:,i + size_test/2 + size_test) =  imfilter(I,h1,'replicate');
    test_data(:,:,:,i + size_test/2 + 2*size_test) =  imfilter(I,h2,'replicate');
    test_data(:,:,:,i + size_test/2 + 3*size_test) =  imfilter(I,h3,'replicate');
end
train_label_p = [ones(size_train/2,1); zeros(size_train/2,1); 0.9*ones(size_train/2,1); 0.1*ones(size_train/2,1); 
    0.7*ones(size_train/2,1); 0.2*ones(size_train/2,1);
    0.5*ones(size_train/2,1); 0.3*ones(size_train/2,1);];
test_label_p = [ones(size_test/2,1); zeros(size_test/2,1); 0.9*ones(size_test/2,1); 0.1*ones(size_test/2,1); 
    0.7*ones(size_test/2,1); 0.2*ones(size_test/2,1);
    0.5*ones(size_test/2,1); 0.3*ones(size_test/2,1);];

TNTaskbased = zeros(1,5);
FPTaskbased = zeros(1,5);
TNTaskThreshold = zeros(1,5);
FPTaskThreshold = zeros(1,5);

Threshold = [0.899 0.699 0.499 0.299 0.099];
%% Threshold = 0.899
load('net_t0.899.mat');
threshold = 0.899;
train_labels = (train_label_p > threshold);
test_labels = (test_label_p > threshold);
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
TestPred = classify(net,test_data);
TestAccuracy = sum(TestPred == test_labels)/numel(test_labels);
C = confusionmat(test_labels,TestPred);
TNTaskbased(1,1) = C(2,1)/(C(2,1)+C(2,2));
FPTaskbased(1,1) = C(1,2)/(C(1,1)+C(1,2));
load('net.mat');
pre_p = predict(net,test_data/255);
TestPred = (pre_p > threshold) ;
Test_label = (test_label_p > threshold);
TestAccuracy = sum(TestPred == Test_label)/4000;
C = confusionmat(Test_label,TestPred);
TNTaskThreshold(1,1) = C(2,1)/(C(2,1)+C(2,2));
FPTaskThreshold(1,1) = C(1,2)/(C(1,1)+C(1,2));

%% Threshold = 0.699
load('net_t0.699.mat');
threshold = 0.699;
train_labels = (train_label_p > threshold);
test_labels = (test_label_p > threshold);
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
TestPred = classify(net,test_data);
TestAccuracy = sum(TestPred == test_labels)/numel(test_labels);
C = confusionmat(test_labels,TestPred);
TNTaskbased(1,2) = C(2,1)/(C(2,1)+C(2,2));
FPTaskbased(1,2) = C(1,2)/(C(1,1)+C(1,2));
load('net.mat');
pre_p = predict(net,test_data/255);
TestPred = (pre_p > threshold) ;
Test_label = (test_label_p > threshold);
TestAccuracy = sum(TestPred == Test_label)/4000;
C = confusionmat(Test_label,TestPred);
TNTaskThreshold(1,2) = C(2,1)/(C(2,1)+C(2,2));
FPTaskThreshold(1,2) = C(1,2)/(C(1,1)+C(1,2));

%% Threshold = 0.499
load('net_t0.499.mat');
threshold = 0.499;
train_labels = (train_label_p > threshold);
test_labels = (test_label_p > threshold);
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
TestPred = classify(net,test_data);
TestAccuracy = sum(TestPred == test_labels)/numel(test_labels);
C = confusionmat(test_labels,TestPred);
TNTaskbased(1,3) = C(2,1)/(C(2,1)+C(2,2));
FPTaskbased(1,3) = C(1,2)/(C(1,1)+C(1,2));
load('net.mat');
pre_p = predict(net,test_data/255);
TestPred = (pre_p > threshold) ;
Test_label = (test_label_p > threshold);
TestAccuracy = sum(TestPred == Test_label)/4000;
C = confusionmat(Test_label,TestPred);
TNTaskThreshold(1,3) = C(2,1)/(C(2,1)+C(2,2));
FPTaskThreshold(1,3) = C(1,2)/(C(1,1)+C(1,2));



%% Threshold = 0.299
load('net_t0.299.mat');
threshold = 0.299;
train_labels = (train_label_p > threshold);
test_labels = (test_label_p > threshold);
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
TestPred = classify(net,test_data);
TestAccuracy = sum(TestPred == test_labels)/numel(test_labels);
C = confusionmat(test_labels,TestPred);
TNTaskbased(1,4) = C(2,1)/(C(2,1)+C(2,2));
FPTaskbased(1,4) = C(1,2)/(C(1,1)+C(1,2));
load('net.mat');
pre_p = predict(net,test_data/255);
TestPred = (pre_p > threshold) ;
Test_label = (test_label_p > threshold);
TestAccuracy = sum(TestPred == Test_label)/4000;
C = confusionmat(Test_label,TestPred);
TNTaskThreshold(1,4) = C(2,1)/(C(2,1)+C(2,2));
FPTaskThreshold(1,4) = C(1,2)/(C(1,1)+C(1,2));

%% Threshold = 0.099
load('net_t0.099.mat');
threshold = 0.099;
train_labels = (train_label_p > threshold);
test_labels = (test_label_p > threshold);
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
TestPred = classify(net,test_data);
TestAccuracy = sum(TestPred == test_labels)/numel(test_labels);
C = confusionmat(test_labels,TestPred);
TNTaskbased(1,5) = C(2,1)/(C(2,1)+C(2,2));
FPTaskbased(1,5) = C(1,2)/(C(1,1)+C(1,2));

load('net.mat');
pre_p = predict(net,test_data/255);
TestPred = (pre_p > threshold) ;
Test_label = (test_label_p > threshold);
TestAccuracy = sum(TestPred == Test_label)/4000;
C = confusionmat(Test_label,TestPred);
TNTaskThreshold(1,5) = C(2,1)/(C(2,1)+C(2,2));
FPTaskThreshold(1,5) = C(1,2)/(C(1,1)+C(1,2));

TN = [TNTaskbased;TNTaskThreshold];
FP = [FPTaskbased;FPTaskThreshold];
bar(Threshold,TN')
legend('Task Based','Probabilistic Classification');
xlabel('Treshold');
ylabel('True Negative Rate');
figure,bar(Threshold,FP')
legend('Task Based','Probabilistic Classification');
xlabel('Treshold');
ylabel('False Positive Rate');






