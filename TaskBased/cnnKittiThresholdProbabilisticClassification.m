clear all;
close all;

%% load data


train_images = [];
train_labels = [];
root_dir = './';

pedresize_dir = fullfile(root_dir, '/ped');
other_dir = fullfile(root_dir,'/non-ped');






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

train_labels = train_label_p;
test_labels = test_label_p;

train_data = train_data/255;
test_data = test_data/255;

%% set hyper parameter
PoolingSize =  3;
FilterSize =  5;
Units =  32;
LearningRate =  0.05;
MomentumRate = 0.7 ;
InputScale =  false;


layers = [
    imageInputLayer([height width channel])

    
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
    
    
    fullyConnectedLayer(1)
    sigmoidLayer
    regressionLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',LearningRate, ...
    'MaxEpochs',50, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',5, ...
    'ValidationPatience',5, ...
    'ExecutionEnvironment','gpu', ...
    'Verbose',true, ...
    'Plots','none',...
    'ResetInputNormalization',InputScale);



%% train network
[net info] = trainNetwork(train_data,train_labels,layers,options);


%% save model
save('net.mat','net');



