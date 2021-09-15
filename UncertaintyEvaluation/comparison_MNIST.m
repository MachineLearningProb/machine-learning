clear all;
close all;

%% load data
[train_images, train_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
train_labels = categorical(uint8(train_labels));
test_labels = categorical(uint8(test_labels));
size_train = 60000;
size_test = 10000;
num_class = 10;
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

%% load pretrained net
sam_num = 1000;
load('CNN_MNIST_net.mat');
cnn_net = net;
load('resNet_MNIST_net.mat');
resnet = net;


%% create sub dataset
testerror = zeros(10,sam_num);
sample_rate = 0.05;
accumulte_num = zeros(10,1);
for sample_num = 1:sam_num
    %sample_num
    index = randperm(10000,10000*sample_rate);
    subtest_data = test_data(:,:,1,index);
    subtest_labels = test_labels(index,1);
    LabelTest = uint8(subtest_labels);




    cnn_current_results = cnn_net.activations(subtest_data(:,:,1,1:10000*sample_rate),'softmax');
    cnn_predict_results = sum(cnn_current_results,4);
    cnn_predict_results = squeeze(cnn_predict_results);
    resnet_current_results = resnet.activations(subtest_data(:,:,1,1:10000*sample_rate),'softmax');
    resnet_predict_results = sum(resnet_current_results,4);
    resnet_predict_results = squeeze(resnet_predict_results);
    
    true_results = tabulate(LabelTest);
    true_results = true_results(:,2);
    cnn_error(:,sample_num) = abs(cnn_predict_results - true_results);
    resnet_error(:,sample_num) = abs(resnet_predict_results - true_results);
end

%% Compute Brier Score
cnn_TestPred = classify(cnn_net,test_data);
cnn_TestAccuracy = sum(cnn_TestPred == test_labels)/numel(test_labels);
cnn_test_results = cnn_net.activations(test_data,'softmax');

cnn_BS = computeBS(double(test_labels'),cnn_test_results);
cnn_ECE = computeECE(double(test_labels'),cnn_test_results,double(cnn_TestPred));

resnet_TestPred = classify(resnet,test_data);
resnet_TestAccuracy = sum(resnet_TestPred == test_labels)/numel(test_labels);
resnet_test_results = resnet.activations(test_data,'softmax');

resnet_BS = computeBS(double(test_labels'),resnet_test_results);
resnet_ECE = computeECE(double(test_labels'),resnet_test_results,double(resnet_TestPred));

cnn_error = cnn_error/50;
resnet_error = resnet_error/50;

%% Compute sorted results
sortedP  = [];
P = squeeze(cnn_test_results);
for class = 1:10
    index = find(double(test_labels) == class);
    PwithIndex = [P(:,index); index'];
    sortedP =[sortedP PwithIndex];
end
[maxP index]= max(sortedP(1:10,:));
sortedResult = [maxP; index];
correctLabel = [];
for class = 1:10
    num = length(find(double(test_labels) == class));
    correctLabel = [correctLabel class*ones(1,num)];
end

cnn_sortedResult = [sortedResult; correctLabel;sortedP(11,:)];



sortedP  = [];
P = squeeze(resnet_test_results);
for class = 1:10
    index = find(double(test_labels) == class);
    PwithIndex = [P(:,index); index'];
    sortedP =[sortedP PwithIndex];
end
[maxP index]= max(sortedP(1:10,:));
sortedResult = [maxP; index];
correctLabel = [];
for class = 1:10
    num = length(find(double(test_labels) == class));
    correctLabel = [correctLabel class*ones(1,num)];
end

resnet_sortedResult = [sortedResult; correctLabel;sortedP(11,:)];


%% Compute sub test data mean and variance of error
cnn_mean_error = sum(cnn_error,2)/sample_num;
cnn_diff = cnn_error - repmat(cnn_mean_error,1,sample_num);
cnn_var_error = sum(cnn_diff.*cnn_diff,2)/sample_num;
cnn_result_mean_var = [cnn_mean_error cnn_var_error];

resnet_mean_error = sum(resnet_error,2)/sample_num;
resnet_diff = resnet_error - repmat(resnet_mean_error,1,sample_num);
resnet_var_error = sum(resnet_diff.*resnet_diff,2)/sample_num;
resnet_result_mean_var = [resnet_mean_error resnet_var_error];

final_results = [cnn_result_mean_var resnet_result_mean_var];

%% Plot Results

% C_cnn = confusionmat(test_labels,cnn_TestPred);
% confusionchart(C_cnn);
% figure;
% C_resnet = confusionmat(test_labels,resnet_TestPred);
% confusionchart(C_resnet);

class = cell(20000,1);
for i = 1:10
    class(1+(i-1)*1000:i*1000) = {num2str(i-1)};
    class(1+(i-1)*1000+10000:i*1000+10000) = {num2str(i-1)};
end

method = cell(20000,1);
method(1:10000) = {'cnn'};
method(10001:20000) = {'resnet'};
error = [reshape(cnn_error',1,10000) reshape(resnet_error',1,10000)];
error = error;
error = error';
tbl = table(method,class,error);
tbl.class = categorical(tbl.class);
boxchart(tbl.class,tbl.error,'GroupByColor',tbl.method,'MarkerStyle','none');
xlabel('Class');
ylabel('Error');
hold on;
x1 = 0.75:1:9.75;
x2 = 1.25:1:10.25;
plot(x1,final_results(:,1),'-o');
plot(x2,final_results(:,3),'-o');
legend('cnn','resnet','cnn mean','resnet mean');

figure;
class = categorical({'0','1','2','3','4','5','6','7','8','9'});
class = reordercats(class,{'0','1','2','3','4','5','6','7','8','9'});
bar(class,[cnn_var_error';resnet_var_error']);
xlabel('Class');
ylabel('Variance of Error');
legend('cnn','resnet');


