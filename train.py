import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
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
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(10, activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            #model.fit(X_train, Y_train , epoch=200, batch_size=32, verbose=1)
            #model.save('saved_models/cifar10_1')
            return model

        elif model=="CIFAR100":
            (X_train, Y_train),(X_test, Y_test) = cifar10.load_data()
            X_trian = X_train.astype('float32')
            X_test = X_test.astype('float32')
            img_width, img_height, img_num_channels = 32, 32, 3
            input_shape = (img_width, img_height, img_num_channels)


            X_train = X_train/255
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(100, activation='softmax'))
            return model
            #model.save('saved_models/cifar10_3')



            

        elif model=="MNIST":
            (X_train, Y_train),(X_test, Y_test) = mnist.load_data()
            model = Sequential()
            model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
            model.add(MaxPooling2D(2,2))
            model.add(Flatten())
            model.add(Dense(100,activation='relu'))
            model.add(Dense(10, activation='softmax'))
            #model.compile(loss='binary_crossentropy', optimizer = 'sgd',metrics=['accuracy'])
            #model.fit(X_train,Y_train, epochs=100, batch_size=32,verbose=0)
            #model.save('saved_models/mnist')
            return model


    def savemodel(self):
        model = ['CIFAR10','CIFAR100','MNIST']
        for i in model:
            model1 = self.CNN(model)
            model.compile(loss='sparse_categorical_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            model.fit(X_train, Y_train , epoch=200, batch_size=32, verbose=1)
            model.save('saved_models/'+i+'1')
            model2 = self.CNN(model)
            model.compile(loss='binary_crossentropy', optimizer = 'sgd', metrics=['accuracy'])
            model.fit(X_train, Y_train , epoch=200, batch_size=32, verbose=1)
            model.save('saved_models/'+i+'2')




        


