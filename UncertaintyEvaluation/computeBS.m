function BS = computeBS(label,predict_result)

size_data = length(label);
I1 = ones(1,size_data);
I2 = ones(1,size_data);
I3 = double(label);
I4 = 1:size_data;
true_label_index = sub2ind(size(predict_result),I1,I2,I3,I4);


firstterm = ones(1,size_data);
secondterm = -2*predict_result(true_label_index);
thirdterm = squeeze(sum(predict_result.*predict_result,3));

BS = sum(firstterm'+secondterm'+thirdterm)/size_data;