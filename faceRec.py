import cv2
import numpy as numpy

faceDect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
while(True):
	rec,image=cam.read()
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces=faceDect.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+y,y+h),(0,255,0),2)
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release()
cv2.destroyAllWindows()
		