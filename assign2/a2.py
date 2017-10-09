import cv2
import numpy as np
vidcap = cv2.VideoCapture('Video_sample.mp4')
success,image = vidcap.read()
count = 0
success = True
while(True):
  success,image = vidcap.read()
  if(count==0):
	  cv2.imwrite("p/frame%d.jpg" % count, image)
	  count += 1
  else:
	  break
vidcap.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture('Video_sample.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()
	refimage=cv2.imread('p/frame%d.jpg'%0)
	surf = cv2.xfeatures2d.SURF_create()
	kp1, des1 = surf.detectAndCompute(refimage,None)
	kp2, des2 = surf.detectAndCompute(frame,None)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])
	img3 = cv2.drawMatchesKnn(refimage,kp1,frame,kp2,good,None,flags=2)
	r = 1000 / img3.shape[1]
	dim = (1000, int(img3.shape[0] * r))
	resized = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)
	cv2.imshow('frame',resized)
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()