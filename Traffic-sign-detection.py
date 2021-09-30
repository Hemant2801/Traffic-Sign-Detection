#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import pickle
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator


# # parameters

# In[2]:


path = 'Train'
labelsfile = 'labels.csv'
batch_size_val = 50
per_epoch_steps = 502
epochs_val =10
image_dim = (32,32,3)
test_ratio = 0.2
val_ratio = 0.2


# # importing all the images

# In[3]:


count = 0
images= []
classNo = []
mylist = os.listdir(path)
print('TOTAL CLASS DETECTED :',len(mylist))
total_classes = len(mylist)
print('IMPORTING CLASSES..............................')
for i in range(0 , len(mylist)):
    mypiclist = os.listdir(path+'/'+str(count))
    for j in mypiclist:
        img = Image.open(path+'/'+str(count)+'/'+str(j))
        img = img.resize((32,32))
        currImg = np.array(img)
        images.append(currImg)
        classNo.append(count)
    print(count,end = ' ')
    count += 1


# In[4]:


print(' ')
images = np.array(images)
classNo = np.array(classNo)
print(images.shape)
print(classNo.shape)


# # split data

# In[5]:


x_train , x_test , y_train , y_test = train_test_split(images, classNo , test_size = test_ratio)
x_train , x_val , y_train , y_val = train_test_split(x_train , y_train , test_size = val_ratio)
#x_train = array of images to train
#y_train = corresponding class ID


# # to check that no. of images matches the labels for each data set

# In[6]:


print('DATA SHAPE')
print('TRAIN:',end= ' ');print(x_train.shape,y_train.shape)
print('VALIDATION:',end= ' ');print(x_val.shape,y_val.shape)
print('TEST:',end= ' ');print(x_test.shape,y_test.shape)


# # reading the labels

# In[7]:


data = pd.read_csv(labelsfile)
print(data.shape)


# # displaying some sample images from the data

# In[8]:


samples_no = []
cols =5
num_classes = total_classes
fig , axis = plt.subplots(nrows=num_classes , ncols=cols , figsize = (5,300))
fig.tight_layout()
for i in range(cols):
    for index ,rows in data.iterrows():
        x_selected = x_train[y_train==index]
        axis[index][i].imshow(x_selected[random.randint(0,len(x_selected)-1),:,:] , cmap = plt.get_cmap('gray'))
        axis[index][i].axis('off')
        if i == 2:
            axis[index][i].set_title(str(index)+'-'+rows['names'])
            samples_no.append(len(x_selected))


# # displaying a bar chart
# 

# In[9]:


print(samples_no)
plt.figure(figsize=(12,4))
plt.bar(range(0 , num_classes) ,samples_no)
plt.title('DISTRIBUTION OF TRAINING DATASET')
plt.xlabel('CLASS NUMBER')
plt.ylabel('NUMBER OF IMAGES')
plt.show()


# # Preprocessing the images

# In[10]:


def grayscale(img):
    img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)  #converts the image into gray
    img = equalize(img)   #standardize the lighting of the image
    img = img/255         #to normalize values between 0 and 1 instead of 0 to 255
    return img
    
x_train = np.array(list(map(preprocessing ,x_train))) #to iterate through all images and preprocess the images
x_val = np.array(list(map(preprocessing ,x_val)))
x_test = np.array(list(map(preprocessing ,x_test)))

cv2.imshow('Gray Scale images', x_train[random.randint(0,len(x_train)-1)]) #to check if the training is done properly
cv2.waitKey(0)


# # Add a depth of 1

# In[11]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)


# # Augmentation of images : to make them more generic

# In[12]:


datagen = ImageDataGenerator(width_shift_range=0.1,  #0.1 = 10% horizontal shift
                             height_shift_range=0.1,  #vertical shift
                             zoom_range=0.2,  #0.2 means zoom range is 1 +- 0.2
                             shear_range=0.1,  #magnitude of shear angle
                             rotation_range=10 #degrees
                            )

datagen.fit(x_train)
batches = datagen.flow(x_train ,y_train , batch_size=20)  #requests datagen togenerates the images #batch size = no. of images generated each time its called
x_batch , y_batch = next(batches)


# # to show augmented image samples

# In[13]:


fig , axs = plt.subplots(1,15,figsize=(20,5))
fig.tight_layout
for i in range(15):
    axs[i].imshow(x_batch[i].reshape(image_dim[0] , image_dim[1]))
    axs[i].axis('off')
plt.show()


# In[14]:


y_train = to_categorical(y_train,total_classes)
y_val = to_categorical(y_val,total_classes)
y_test = to_categorical(y_test,total_classes)


# # Convolutional neural network model

# In[15]:


def mymodel():
    filter_no = 60
    filter1_size = (5,5) #this is the kernel that moves around the image to get the features
                        #this will remove 2 piece from the each border when using (32,32) image
    filter2_size = (3,3)
    poolsize = (2,2)   #scale down all features map generalizing more , reducws overfitting
    nodes_count = 500  #total no of nodes in hidden layer
    model = Sequential()
    model.add(Conv2D(filter_no,filter1_size , input_shape = (image_dim[0],image_dim[1],1) , activation ='relu')) #adding more convolutional layer to the model
    model.add(Conv2D(filter_no ,filter1_size,activation = 'relu'))
    model.add(MaxPool2D(pool_size = poolsize)) #does not affect the no of filters
    
    model.add(Conv2D(filter_no//2,filter2_size,activation='relu'))
    model.add(Conv2D(filter_no//2,filter2_size,activation='relu'))
    model.add(MaxPool2D(pool_size = poolsize)) 
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(nodes_count,activation='relu'))
    model.add(Dropout(0.5))  #input nodes dropped with each update 1 for all and 0 for None
    model.add(Dense(total_classes, activation ='softmax'))  #output layer
    
    #compiling the model
    model.compile(Adam(lr =0.001) , loss ='categorical_crossentropy' , metrics =['accuracy'])
    return model


# # training the model

# In[17]:


model =mymodel()
print(model.summary())
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size_val), 
                              steps_per_epoch = per_epoch_steps, 
                              epochs = epochs_val,
                              validation_data=(x_val, y_val),
                              shuffle = 1)


# # plot

# In[20]:


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training' ,'validation'])
plt.title('LOSS')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training' ,'validation'])
plt.title('ACCURACY')
plt.xlabel('epoch')


# # storing the object as a joblib object

# In[33]:


model.save('model_trained.h5')


# In[ ]:




