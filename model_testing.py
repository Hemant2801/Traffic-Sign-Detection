#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from keras.models import load_model


# In[2]:


framewidth = 640
frameheight = 480
brightness =100
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX


# # setup the camers

# In[3]:


vid = cv2.VideoCapture(0)
if not vid.isOpened():
    vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print('ERROR!!!!!!!!!!!!874574569890NJHFKBNM,BKMJIT674@#!$&*%^%#%^@#%^$&^')
    
vid.set(3,framewidth)
vid.set(4,frameheight)
vid.set(10 ,brightness)


# # importing the training module

# In[4]:


model = load_model('model_trained.h5')


# # Preprocessing the image

# In[5]:


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


# In[6]:


ls =[]
with open("sign_labels.txt", "r") as file:
    data = file.readlines()
    for i in range(len(data)):
        ls.append(data[i])
ls  = np.asarray(ls).astype('object')
print(ls[4])
def getclassname(classNo):
    return (ls[classNo])


# # rectangular box around the object

# In[11]:


def contour(path):
    img = cv2.resize(path , (640,480))
    backup= img.copy()


    #img = cv2.resize(img,(480,360))
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # red color boundaries [B, G, R]
    lower = [np.mean(image[:,:,i] - np.std(image[:,:,i])/3 ) for i in range(3)]
    upper = [250, 250, 250]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    ret,thresh = cv2.threshold(mask, 40, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    return contours,hierarchy

#cv2.imshow('output' , output)
#cv2.waitKey()


# In[18]:


choice = input('WHAT DO YOU WANT TO DO?\na)VIDEO\nb)IMAGE\n')

if choice == 'a':
    while True:
    #read image
        success , imageread = vid.read()
    #process image
        img = np.array(imageread)
        img = cv2.resize(img,(32,32))
        img = preprocessing(img)
    #cv2.imshow('processed image' , img)
        img = img.reshape(1,32,32,1)
        cv2.putText(imageread, 'CLASS : ', (20,35) , font, 0.75,(0,150,255) , 2, cv2.LINE_AA )
        cv2.putText(imageread, 'PROBABILITY : ', (20,75) , font, 0.75,(0,0,255) , 2, cv2.LINE_AA )
    
    #predict image
        pred = model.predict(img)
        classindex = model.predict_classes(img)
        probability_values = np.amax(pred)
        
        #cont , heir = contour()
        if probability_values > threshold:
            if len(cont) != 0:
                # find the biggest countour (c) by the area
                c = max(cont, key = cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)

                # draw the biggest contour (c) in green
                cv2.rectangle(imageread,(x,y),(x+w,y+h),(0,255,0),5)
                cv2.putText(imageread,str(classindex)+' '+str((getclassname(classindex))),(120,35) , font, 0.75,(0,150,255) , 2, cv2.LINE_AA )
                cv2.putText(imageread,str(round(probability_values*100,2) )+'%', (100,75) , font, 0.75,(0,0,255) , 2, cv2.LINE_AA )
        cv2.imshow('RESULT' , imageread)
    
        if cv2.waitKey(2) & 0xff == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
elif choice == 'b':
    imageread = cv2.imread('speed-50.jpg')
    #process image
    imageread = cv2.resize(imageread,(640,480))
    img = np.array(imageread)
    img = cv2.resize(img,(32,32))
    img = preprocessing(img)
    #cv2.imshow('processed image' , img)
    img = img.reshape(1,32,32,1)
    cv2.putText(imageread, 'CLASS : ', (20,35) , font, 0.75,(0,0,255) , 2, cv2.LINE_AA )
    cv2.putText(imageread, 'PROBABILITY : ', (20,75) , font, 0.75,(0,0,255) , 2, cv2.LINE_AA )
    
    #predict image
    pred = model.predict(img)
    classindex = model.predict_classes(img)

    probability_values = np.amax(pred)

    cont , heir = contour(imageread)
    if probability_values > threshold:
        if len(cont) != 0:
            # find the biggest countour (c) by the area
            c = max(cont, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)

            # draw the biggest contour (c) in green
            cv2.rectangle(imageread,(x,y),(x+w,y+h),(0,255,0),5)
            cv2.putText(imageread,str(classindex)+' '+str((getclassname(classindex))),(120,35) , font, 0.75,(0,0,255) , 2, cv2.LINE_AA )
            cv2.putText(imageread,str(round(probability_values*100,2) )+'%', (200,75) , font, 0.75,(0,0,255) , 2, cv2.LINE_AA )
    cv2.imshow('RESULT' , imageread)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
else:
    print('WRONG KEY PRESSED @E^%@&^*$&*%@$*&%&^@%*^$&%@&$&^%@^$@&^$@&')


# In[ ]:




