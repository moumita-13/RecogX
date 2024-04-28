import cv2
import numpy as np
import face_recognition

imgSrk= face_recognition.load_image_file('Images/Shah Rukh Khan.jpg')
imgSrk=cv2.cvtColor(imgSrk,cv2.COLOR_BGR2RGB)
imgTest= face_recognition.load_image_file('Images/Shah Rukh Khan.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

facloc=face_recognition.face_locations(imgSrk)[0]
encodeSrk=face_recognition.face_encodings(imgSrk)[0]
cv2.rectangle(imgSrk,(facloc[3],facloc[0]),(facloc[1],facloc[2]),(255,0,255),3)

faclocTest=face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faclocTest[3],faclocTest[0]),(faclocTest[1],faclocTest[2]),(255,0,255),3)

results=face_recognition.compare_faces([encodeSrk],encodeTest)
faceDis=face_recognition.face_distance([encodeSrk],encodeTest)
#print(faceDis,results)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('Shah Rukh Khan',imgSrk)
cv2.imshow('SRK Test',imgTest)
cv2.waitKey(0)

