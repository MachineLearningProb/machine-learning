import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import random


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
#Y_train = generate(Y_train)
#Y_test = generate(Y_test)
label = Y_test



def CNN(x_train,y_train):
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer = 'sgd',metrics=['accuracy'])
    model.fit(x_train,y_train, epochs=100, batch_size=32,verbose=0)
    return model
    
    
def test_set(x_test, y_test, model, label):
    


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



def RandomSample(X_train, Y_train, X_test,Y_test, label):
    perc = 0.1
    model = CNN(X_train, Y_train)
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
        accuracy,true,threshold =test_set(x_test, y_test, model, label)
        SaveToFile(filename,x_test,y_test,accuracy,true, threshold, X_test.shape[0])
def SaveToFile(filename, x_train, y_train, accuracy, true, thres,le):
    data = open(filename+'.txt','w')
    result = open(filename+'res.txt','w')

    for i in range(len(x_train)):
        data.write(str(x_train[i]))
    for i in range(len(y_train)):
        data.write(str(y_train[i])+'\n')
        result.write("Total number of ture values found in the subset of the test set ="+str(true)+'\n')
    for i in range(len(accuracy)):
        result.write("labels predicted =" +str(accuracy[i])+" total = " +str(le)+" threshold = " +str(thres)+'\n')

    #result.write("accuracy = "+str(accuracy)+" total = " +str(le)+" percentage = "+str(accuracy\le)+"threshold = "+str(thres)+"true = "+str(true)+'\n')
    #result.write("accuracy = " +str(accuracy)+" total = "+str(le)+" precentage = "+str(accuracy/le)+" threshold = "+str(thres)+" true = "+str(true)+'\n')
    data.close()
    result.close()
        


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

Y_train = generate(Y_train)
Y_test = generate(Y_test)
RandomSample(X_train, Y_train, X_test, Y_test,label)



