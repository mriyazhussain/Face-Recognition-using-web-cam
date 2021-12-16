#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2, sys, numpy, os


# In[2]:


haar_file = 'haarcascade_frontalface_alt.xml'


# In[3]:


datasets = 'datasets' 


# In[4]:


sub_data = 'Riyaz'
path = os.path.join(datasets, sub_data)
print ("path = {}", path)
if not os.path.isdir(path):
    os.mkdir(path)


# In[5]:


# defining the size of images
(width, height) = (130, 100)   
 
#'0' is used for my webcam,
# if you've any other camera
#  attached use '1' like this
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)


# In[6]:


# The program loops until it has 30 images of the face.
count = 1
while count < 3:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s\% s.png' % (path, count), face_resize)
    count += 1
     
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




