

The two folders contain the code of uncertainty evaluation and task-based network parts.

## Uncertainty evaluation 
1. Befroe run the code, please download the dataset(MNIST and CIFAR10), then put the data with code in the same folder.
2. To test the uncertainty on trained model, run the file comparison_CIFAR10.m and comparison_MNIST.m.
3. There are four trained network CNN_CIFAR10_net.mat, CNN_MNIST_net.mat, resNet_CIFAR10_net.mat and resNet_MNIST_net.mat. To retrain the network, run the code train_CNN_CIFAR10.m
, train_CNN_MNIST.m, train_resNet_CIFAR10.mat and train_resNet_MNIST.mat.

## TaskBased Network
1. Before run the code, unzip the data ped.zip and non-ped.zip with code insame folder which include the training and test images cropped from KITTI dataset.
2. To train the task-based network with different level of treshold, run cnnKittiThresholdTaskBased.m.
3. To train the ordinary probabilistic classification network, run cnnKittiThresholdProbabilisticClassification.m
4. To do the comparison between two networks, run comparisonTaskBasedAndProbabilisticClassification.m


