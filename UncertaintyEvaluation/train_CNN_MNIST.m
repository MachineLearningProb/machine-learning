clear all;
close all;

%% load data


[train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
size_train = 60000;
size_test = 10000;
width = 28;
height = 28;
channel = 1;
train_data = zeros(height,width,channel,size_train);
test_data = zeros(height,width,channel,size_test);
for i = 1:size_train
    train_data(:,:,1,i) = train_images(:,:,i);
end

for i = 1:size_test
    test_data(:,:,1,i) = test_images(:,:,i);
end

%% set hyper parameter
PoolingSize =  3;
FilterSize =  5;
Units =  32;
LearningRate =  0.1;
MomentumRate = 0.7 ;
InputScale =  false;


layers = [
    imageInputLayer([28 28 1])

    
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
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'Momentum' , MomentumRate, ...
    'InitialLearnRate',LearningRate, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',5, ...
    'ValidationPatience',5, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',true, ...
    'Plots','none',...
    'ResetInputNormalization',InputScale);



%% train network
[net info] = trainNetwork(train_data,train_labels,layers,options);

save('CNN_MNIST_net.mat','net');

