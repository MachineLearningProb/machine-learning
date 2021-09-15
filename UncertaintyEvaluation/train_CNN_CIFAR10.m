clear all;
close all;

%% load data


train_images = [];
train_labels = [];

for i = 1:5
    load(strcat('.\data_batch_', num2str(i) ,'.mat'));  
    train_images = [train_images;data];
    train_labels = [train_labels;labels];
end
load('.\test_batch.mat')
test_labels = labels;
test_images = data;

train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));



size_train = 50000;
size_test = 10000;
width = 32;
height = 32;
channel = 3;
train_data = zeros(height,width,channel,size_train);
test_data = zeros(height,width,channel,size_test);
for i = 1:size_train
    train_data(:,:,1,i) = reshape(train_images(i,1:width*height),[width,height]);
    train_data(:,:,2,i) = reshape(train_images(i,width*height+1:2*width*height),[width,height]);
    train_data(:,:,3,i) = reshape(train_images(i,2*width*height+1:3*width*height),[width,height]);
end

for i = 1:size_test
    test_data(:,:,1,i) = reshape(test_images(i,1:width*height),[width,height]);
    test_data(:,:,2,i) = reshape(test_images(i,width*height+1:2*width*height),[width,height]);
    test_data(:,:,3,i) = reshape(test_images(i,2*width*height+1:3*width*height),[width,height]);
end

%% set hyper parameter
PoolingSize =  3;
FilterSize =  5;
Units =  32;
LearningRate =  0.1;
MomentumRate = 0.7 ;
InputScale =  false;


layers = [
    imageInputLayer([width height channel])

    
    convolution2dLayer(FilterSize,Units,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(PoolingSize,'Stride',2)
    
    convolution2dLayer(FilterSize,2*Units,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(PoolingSize,'Stride',2)
    
    convolution2dLayer(FilterSize,4*Units,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(FilterSize,4*Units,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'Momentum' , MomentumRate, ...
    'InitialLearnRate',LearningRate, ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',5, ...
    'ValidationPatience',5, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',true, ...
    'Plots','none',...
    'ResetInputNormalization',InputScale);



%% train network
[net info] = trainNetwork(train_data,train_labels,layers,options);

save('CNN_CIFAR10_net.mat','net')

