import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='Images'
images=[]
clNames=[]
dataList=os.listdir(path)
#print(dataList)

for cl in dataList:
    currentImg=cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    clNames.append(os.path.splitext(cl)[0])
print(clNames)

def doEncoding(images):
    encodedList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoded=face_recognition.face_encodings(img)[0]
        encodedList.append(encoded)
    return encodedList

def markAttendance(name):
    now = datetime.now()
    currentDate = now.strftime('%d-%m-%Y')
    filename = f'Record/{currentDate}.csv'
    if not os.path.isfile(filename):
        with open(filename, 'w') as f:
            f.write('Name,Date,Time')

    with open(filename, 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            datString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{datString},{timeString}')
    
encodeListKnown=doEncoding(images)
# print(len(encodeListKnown))
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    smallImg=cv2.resize(img,(0,0),None,0.25,0.25)
    smallImg=cv2.cvtColor(smallImg,cv2.COLOR_BGR2RGB)
    
    CurFrame=face_recognition.face_locations(smallImg)
    encodeCFrame=face_recognition.face_encodings(smallImg,CurFrame)
    
    for encodeFace,faceLoct in zip(encodeCFrame,CurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist=face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDist)
        matchIndex=np.argmin(faceDist)
        
        if matches[matchIndex]:
            name=clNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1=faceLoct
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)
    
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)