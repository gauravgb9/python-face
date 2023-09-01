# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:01:23 2022

@author: rupik
"""
import cv2
import numpy as np

vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/rupik/Documents/Edu labs/haarcascade_frontalface_default.xml')


skip =0
face_data = []

#create a folder tp store all image samples
database_path = "C:/Users/rupik/Documents/Edu labs/faces/"

#enter name of person
file_name = input('Enter the name of person: ')

while True:
    ret,frame=vid.read()
    if ret == True:
        frame = cv2.resize(frame,(1280,960),fx=0,fy=0,
                           interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
        
    if len(faces)==0:
           continue
    k=1
        
    faces = sorted(faces,key = lambda x:x[2]*x[3], reverse= False)
    skip+=1
        
        
    for face in faces[:1]:
            x,y,w,h = face
            offset = 5
            face_offset = frame[y-offset:y+offset,x-offset:x+offset]
            face_selection = cv2.resize(face_offset,(100,100))
        
        
            # to select a picture once every 10 second
            if skip%10==0:
                face_data.append(face_selection)
                print(len(face_data))
                
            cv2.imshow(str(k),face_selection)
            
            k+=1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,155,100),2)
        
                
            
        
    cv2.imshow('f1',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
face_data = np.array(face_data)    
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data)    
#save image in the location
np.save(database_path + file_name,face_data)
print('dataset saved at : {}'.format (database_path + file_name + '.npy'))
# wait for any key
vid.release()
# CLose all the frames
cv2.destroyAllWindows()