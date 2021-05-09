import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist, cifar10, cifar100
import random


class model:
    """Training model cifar10, cifar100, and minist  and saves them into folder"""


    def CNN(self, model):
        if model=="CIFAR10":
            (X_train, Y_train),(X_test, Y_test) = cifar10.load_data()
            X_trian = X_train.astype('float32')
            X_test = X_test.astype('float32')
            img_width, img_height, img_num_channels = 32, 32, 3
            input_shape = (img_width, img_height, img_num_channels)


            X_train = X_train/255
            
            #model 1
            modelA = Sequential()
            modelA.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            modelA.add(MaxPooling2D(pool_size=(2, 2)))
            modelA.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            modelA.add(MaxPooling2D(pool_size=(2, 2)))
            modelA.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            modelA.add(MaxPooling2D(pool_size=(2, 2)))
            modelA.add(Flatten())
            modelA.add(Dense(256, activation='relu'))
            modelA.add(Dense(128, activation='relu'))
            modelA.add(Dense(10, activation='softmax'))
            modelA.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            modelA.fit(X_train, Y_train , epochs=200, batch_size=16, verbose=1)
            modelA.save('saved_models/cifar10_1')
            
            #model 2
            modelB = Sequential()
            modelB.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            modelB.add(MaxPooling2D(pool_size=(2, 2)))
            modelB.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            modelB.add(MaxPooling2D(pool_size=(2, 2)))
            modelB.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            modelB.add(MaxPooling2D(pool_size=(2, 2)))
            modelB.add(Flatten())
            modelB.add(Dense(256, activation='relu'))
            modelB.add(Dense(128, activation='relu'))
            modelB.add(Dense(10, activation='softmax'))
            modelB.compile(loss='binary_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            modelB.fit(X_train, Y_train , epochs=200, batch_size=16, verbose=1)
            modelB.save('saved_models/cifar10_2')

        elif model=="CIFAR100":
            (X_train, Y_train),(X_test, Y_test) = cifar10.load_data()
            X_trian = X_train.astype('float32')
            X_test = X_test.astype('float32')
            img_width, img_height, img_num_channels = 32, 32, 3
            input_shape = (img_width, img_height, img_num_channels)


            X_train = X_train/255

            #model 1
            modelA = Sequential()
            modelA.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            modelA.add(MaxPooling2D(pool_size=(2, 2)))
            modelA.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            modelA.add(MaxPooling2D(pool_size=(2, 2)))
            modelA.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            modelA.add(MaxPooling2D(pool_size=(2, 2)))
            modelA.add(Flatten())
            modelA.add(Dense(256, activation='relu'))
            modelA.add(Dense(128, activation='relu'))
            modelA.add(Dense(100, activation='softmax'))
            modelA.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            modelA.fit(X_train, Y_train , epochs=200, batch_size=16, verbose=1)
            modelA.save('saved_models/cifar100_1')

            
            #return model
            #model.save('saved_models/cifar10_3')
            
            #model 2
            modelB = Sequential()
            modelB.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            modelB.add(MaxPooling2D(pool_size=(2, 2)))
            modelB.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            modelB.add(MaxPooling2D(pool_size=(2, 2)))
            modelB.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            modelB.add(MaxPooling2D(pool_size=(2, 2)))
            modelB.add(Flatten())
            modelB.add(Dense(256, activation='relu'))
            modelB.add(Dense(128, activation='relu'))
            modelB.add(Dense(100, activation='softmax'))
            modelB.compile(loss='binary_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            modelB.fit(X_train, Y_train , epochs=200, batch_size=16, verbose=1)
            modelB.save('saved_models/cifar100_2')




            

        elif model=="MNIST":
            

            (X_train, Y_train),(X_test, Y_test) = mnist.load_data()
            
            #model 1
            model = Sequential()
            model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
            model.add(MaxPooling2D(2,2))
            model.add(Flatten())
            model.add(Dense(100,activation='relu'))
            model.add(Dense(10, activation='softmax'))
            #model.compile(loss='binary_crossentropy', optimizer = 'sgd',metrics=['accuracy'])
            #model.fit(X_train,Y_train, epochs=100, batch_size=32,verbose=0)
            #model.save('saved_models/mnist')
            modelA.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            modelA.fit(X_train, Y_train , epochs=200, batch_size=16, verbose=1)
            modelA.save('saved_models/MNIST_1')

            
            #model 2
            model = Sequential()
            model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
            model.add(MaxPooling2D(2,2))
            model.add(Flatten())
            model.add(Dense(100,activation='relu'))
            model.add(Dense(10, activation='softmax'))
            modelB.compile(loss='binary_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            modelB.fit(X_train, Y_train , epochs=200, batch_size=16, verbose=1)
            modelB.save('saved_models/MNIST_2')




    def savemodel(self):
        model = ['CIFAR10','CIFAR100','MNIST']
        for i in model:
            model = self.CNN(i)



if __name__=="__main__":
    

    new = model()
    new.savemodel()
