function ECE = computeECE(label,predict_result,predict_label)
size_data = length(label);
I1 = ones(1,size_data);
I2 = ones(1,size_data);
I3 = double(label);
I4 = 1:size_data;
true_label_index = sub2ind(size(predict_result),I1,I2,I3,I4);
true_label_predict_result = predict_result(true_label_index);
Edge = 0:0.1:1;
numOfBin = length(Edge) - 1;
[N,Edge,Bin] = histcounts(true_label_predict_result,Edge);
ratio = N/size_data;
Indicator = double(label == predict_label');
acc = zeros(numOfBin,1);
conf = zeros(numOfBin,1);
for i  = 1:size_data
    acc(Bin(1,i)) = acc(Bin(1,i))+Indicator(1,i);
    conf(Bin(1,i)) = conf(Bin(1,i))+true_label_predict_result(1,i);
end
ECE_each_bin = abs(acc - conf);
ratioOfEachBin = zeros(1,length(N));
for i = 1:length(N)
    if(N(1,i) == 0)
        ratioOfEachBin(1,i) = 0;
    else
        ratioOfEachBin(1,i) = 1/N(1,i);
    end
end
ECE_each_bin = ECE_each_bin.*ratioOfEachBin';


ECE = sum(ECE_each_bin.*ratio');



