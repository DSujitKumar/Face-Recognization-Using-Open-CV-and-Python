import cv2
import numpy as numpy

faceDect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingData.yml')
id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
while(True):
	rec,image=cam.read()
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	faces=faceDect.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+y,y+h),(0,255,0),2)
		id,config=recognizer.predict(gray[y:y+h,x:x+w])
		
		if (config<70):
			if (id==1):
				id="Sujit"

		else:
			id="Unknown"
		cv2.putText(image, str(id), (x-30,y+25), fontface, fontscale, fontcolor) 
		
	cv2.imshow("face",image)
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release()
cv2.destroyAllWindows()
		