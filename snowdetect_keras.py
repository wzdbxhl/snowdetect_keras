#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:33:30 2017

@author: jchen
"""
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

def img_dir(dir):
    files = []
    filesname = []
    
    for root,dirs,filename in os.walk(dir):
        for file in filename:
            files.append(os.path.join(root,file))
            filesname.append(filename)
    return files

def data_augment(dir_in,dir_out):
    
    files_in = img_dir(dir_in)
    files_out = img_dir(dir_out)
    
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    for i in range(len(files_in)):
        img = load_img(files_in[i])      
        x = img_to_array(img)  #  Numpy array 
        x = x.reshape((1,) + x.shape) 

        n = 0    #  data augmentation to prevent overfitting
        for batch in datagen.flow(x, batch_size=1,save_to_dir='/home/jchen/Documents/pic/data/preview/in', save_prefix='in', save_format='jpeg'):
            n += 1
            if n > 20:
                break  #  to stop the loop 
                
    for j in range(len(files_out)):
        img = load_img(files_out[j])      
        x = img_to_array(img)  #  Numpy array 
        x = x.reshape((1,) + x.shape) 

        n = 0    #  data augmentation to prevent overfitting
        for batch in datagen.flow(x, batch_size=1,save_to_dir='/home/jchen/Documents/pic/data/preview/out', save_prefix='out', save_format='jpeg'):
            n += 1
            if n > 20:
                break  #  to stop the loop    
    return

#dir_in = '/home/jchen/Documents/pic/data/train/in'
#dir_out = '/home/jchen/Documents/pic/data/train/out'
#data_augment(dir_in,dir_out)


img_width, img_height = 224, 224

train_data_dir = '/home/jchen/Documents/pic/data_large/data/train'
validation_data_dir = '/home/jchen/Documents/pic/data_large/data/validation'
#test_data_dir = '/home/jchen/Documents/pic/data_large/data/test'

nb_train_samples = 7500
nb_validation_samples = 2500
nb_test_samples = 922
epochs = 60
batch_size = 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def create_model_1():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='valid', kernel_initializer='glorot_normal', input_shape=input_shape))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))#'relu'
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64, input_dim=64,
                    kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.01)))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])#adagrad

    return model

#def vgg():
#    model = Sequential()
#    model.add(ZeroPadding2D((1,1),input_shape = input_shape))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(128, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(128, (3, 3), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(256, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(256, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(256, (3, 3), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(512, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(512, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(512, (3, 3), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(512, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(512, (3, 3), activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Conv2D(512, (3, 3), activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
#    model.add(Flatten())
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(4096, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(1000, activation='softmax'))  
#    
#    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics=['accuracy'])
#    
#    return model
#


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    #color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle = True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    #color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle = True)

#test_generator = test_datagen.flow_from_directory(
#    test_data_dir,
#    target_size=(img_width, img_height),
#    #color_mode='grayscale',
#    batch_size=batch_size,
#    class_mode='binary',
#    shuffle = True)


early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model = create_model_1()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)
    #callbacks=[early_stopping])

#model.save_weights('/home/jchen/Documents/model/data10000.h5')
#print(history.history.keys())
#
##plot accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
##plot loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#
#
#
#scoreSeg = model.evaluate_generator(validation_generator,nb_validation_samples)
#print("Accuracy = ",scoreSeg[1])
#predict = model.predict_generator(test_generator,nb_test_samples)


################
#img_width, img_height = 150, 150
#
#train_data_dir = '/home/jchen/Documents/pic/data/preview/train'
#validation_data_dir = '/home/jchen/Documents/pic/data/preview/validation'
#
#nb_train_samples = 80
#nb_validation_samples = 24
#
#batch_size = [10, 20, 40, 60, 80, 100]
#epochs = [10, 50, 100]
#param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
#if K.image_data_format() == 'channels_first':
#    input_shape = (3, img_width, img_height)
#else:
#    input_shape = (img_width, img_height, 3)
#
#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Flatten())
##model.add(Dense(64))
#model.add(Dense(64, input_dim=64,
#                kernel_regularizer=regularizers.l2(0.01),
#                activity_regularizer=regularizers.l1(0.01)))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

###########


