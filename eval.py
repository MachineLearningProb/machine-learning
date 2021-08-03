import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import train
from tensorflow.keras.datasets import mnist, cifar10, cifar100
import tensorflow.keras as keras
import random
def normal_dist(x, mean, sd):
    """calculates normal distribution"""
    prob_density = (np.pi*sd) * np.exp(-0.5 * ((x-mean)/sd)**2)
    plt.plot(x,prob_density, color = 'red')
    plt.xlabel('Data Points')
    plt.ylabel('Probability Density')






def marginal_dist():
    """calculates mariginal distribution """

    return 0


def test_set_m1(x_test, y_test, model):

    true=[0,0,0,0,0,0,0,0,0,0]
    accuracy=[0,0,0,0,0,0,0,0,0,0]

    for i in range(len(x_test)):
        y_pre = model.predict(x_test[i].reshape(1,28,28,1))
        test = int(y_test[i])
        #print(test)
        true[test] += 1
        for y in range(len(y_pre[0])):
            accuracy[y]+=y_pre[0][y]
        #print("accuracy = ", accuracy,"y_pred = ", y_pre, "true =", true)
    return accuracy, true


def test_set_m2(x_test, y_test, model, label):



    true=[0,0,0,0,0,0,0,0,0,0]
    accuracy=[0,0,0,0,0,0,0,0,0,0]
    thres = random.random()
    #y_pred = [0,0,0,0,0,0,0,0,0,0]
    for i in range(len(x_test)):
        y_pre = model.predict(x_test[i].reshape(1,28,28,1),y_test[i].any(),verbose=1)
        y_pred = [0,0,0,0,0,0,0,0,0,0]
    #test = int(y_test[i])
    #print(test)
        true[label[i]] += 1
        #true=0
        #print("testing y_pre",y_pre )
        for y in range(len(y_pre[0])):
            if (thres <= y_pre[0][y]):
                y_pred[y]=1
            else:
                y_pred[y] = 0
        y_pred.append("True Value "+str(label[i]))
        accuracy.append(y_pred)
        #print(y_pred)
                #print("true = ",true,"y_pred = ",y_pre[0][y],"thres = ",thres)

    #print("accuracy = ", accuracy,"y_pred = ", y_pre, "true =", true)
    return accuracy, true, thres

def RandomSample(X_test, Y_test):
    perc = 0.1
    if path.exists('saved_models/cifar10_1.h5'):
        c_model1 = keras.models.load_model('saved_models/cifar10_1.h5')
    if path.exists('saved_models/cifar10_2.h5'):
        c_model2 = keras.models.load_model('saved_models/cifar10_2.h5')
    if path.exists('saved_models/MNIST_1.h5'):
        m_model1 = keras.models.load_model('saved_models/MNIST_1.h5')
    if path.exists('saved_models/MNIST_2.h5'):
        m_model2 = keras.models.load_model('saved_models/MNIST_2.h5')
    if path.exists('saved_models/cifar100_1.h5'):
        ch_model1 = keras.models.load_model('saved_models/cifar100_1.h5')
    if path.exists('saved_models/cifar100_2.h5'):
        ch_model2= keras.models.load_model('saved_models/cifar100_2.h5')

    for i in range(100):
        filename = 'sample'
        filename = filename+str(i)
        x_test =[]
        y_test = []
        print("Generating Sample",i)
        if i%10==0:
            perc+=0.1

        for y in range(0,int(perc*X_test.shape[0])):
            Rand = random.randint(0, X_test.shape[0]-1)
            x_test.append(X_test[Rand])
            y_test.append(Y_test[Rand])

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        accuracy,true =test_set_m1(x_test, y_test, m_model1)
        print(accuracy)
        #SaveToFile(filename,x_test,y_test,accuracy, true)



def generate(Y_sample):
    print(len(Y_sample))
    array = []
    for i in range(len(Y_sample)):
        sam = np.zeros(10)
        sam[Y_sample[i]]+=1
        #print(sam, Y_sample[i])
        array.append(sam)
    array = np.array(array)
    return array




if __name__ == '__main__':
    
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],28,28,1)
    X_test = X_test.reshape(X_test.shape[0],28,28,1)
    #Y_train = generate(Y_train)
    #Y_test = generate(Y_test)
    #label = Y_test
    RandomSample(X_test, Y_test)

