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

imageSize = [32 32 3];
pixelRange = [-4 4];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(imageSize,train_data,train_labels, ...
    'DataAugmentation',imageAugmenter, ...
    'OutputSizeMode','randcrop');


%% Define Network Architecture
netWidth = 16;
layers = [
    imageInputLayer([32 32 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
    convolutionalUnit(netWidth,1,'S1U1')
    additionLayer(2,'Name','add11')
    reluLayer('Name','relu11')
    convolutionalUnit(netWidth,1,'S1U2')
    additionLayer(2,'Name','add12')
    reluLayer('Name','relu12')
    
    convolutionalUnit(2*netWidth,2,'S2U1')
    additionLayer(2,'Name','add21')
    reluLayer('Name','relu21')
    convolutionalUnit(2*netWidth,1,'S2U2')
    additionLayer(2,'Name','add22')
    reluLayer('Name','relu22')
    
    convolutionalUnit(4*netWidth,2,'S3U1')
    additionLayer(2,'Name','add31')
    reluLayer('Name','relu31')
    convolutionalUnit(4*netWidth,1,'S3U2')
    additionLayer(2,'Name','add32')
    reluLayer('Name','relu32')
    
    averagePooling2dLayer(8,'Name','globalPool')
    fullyConnectedLayer(10,'Name','fcFinal')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];

lgraph = layerGraph(layers);


lgraph = connectLayers(lgraph,'reluInp','add11/in2');
lgraph = connectLayers(lgraph,'relu11','add12/in2');


skip1 = [
    convolution2dLayer(1,2*netWidth,'Stride',2,'Name','skipConv1')
    batchNormalizationLayer('Name','skipBN1')];
lgraph = addLayers(lgraph,skip1);
lgraph = connectLayers(lgraph,'relu12','skipConv1');
lgraph = connectLayers(lgraph,'skipBN1','add21/in2');

lgraph = connectLayers(lgraph,'relu21','add22/in2');

skip2 = [
    convolution2dLayer(1,4*netWidth,'Stride',2,'Name','skipConv2')
    batchNormalizationLayer('Name','skipBN2')];
lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'relu22','skipConv2');
lgraph = connectLayers(lgraph,'skipBN2','add31/in2');

lgraph = connectLayers(lgraph,'relu31','add32/in2');

plot(lgraph)

numUnits = 9;
netWidth = 16;
lgraph = residualCIFARlgraph(netWidth,numUnits,"standard");


%% Train Network

miniBatchSize = 128;
learnRate = 0.1*miniBatchSize/128;
valFrequency = floor(size(train_data,4)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'InitialLearnRate',learnRate, ...
    'MaxEpochs',80, ...
    'MiniBatchSize',miniBatchSize, ...
    'VerboseFrequency',valFrequency, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationFrequency',valFrequency, ...
    'ExecutionEnvironment','gpu', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',60);

trainedNet = trainNetwork(augimdsTrain,lgraph,options);

net = trainedNet;

save('resNet_CIFAR10_net.mat','net');

