import cv2
import sys
import numpy as np
vidcap = cv2.VideoCapture('v1.mkv')
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

tracker = cv2.TrackerMIL_create()
tracker1 = cv2.TrackerMIL_create()
tracker2 = cv2.TrackerMIL_create()

cap = cv2.VideoCapture('v1.mkv')
while(cap.isOpened()):
	ret, frame = cap.read()
	refimage=cv2.imread('p/frame%d.jpg'%0)
	bbox = cv2.selectROI(refimage, False)
	bbox1 = cv2.selectROI(refimage, False)
	bbox2 = cv2.selectROI(refimage, False)
	ret = tracker.init(frame, bbox)
	ret1 = tracker1.init(frame,bbox1)
	ret2 = tracker2.init(frame,bbox2)
	while True:
		ret,frame = cap.read()
		ret1,frame = cap.read()
		ret2,frame = cap.read()
		if not ret:
			break
		
		ret,bbox=tracker.update(frame)
		ret1,bbox1=tracker1.update(frame)
		ret2,bbox2=tracker2.update(frame)
		surf = cv2.xfeatures2d.SURF_create()
		kp1, des1 = surf.detectAndCompute(frame,None)
		kp2, des2 = surf.detectAndCompute(frame,None)

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)

		flann = cv2.FlannBasedMatcher(index_params, search_params)

		matches = flann.knnMatch(des1,des2,k=2)
		good = []
		for m,n in matches:
			if m.distance < 0.75*n.distance:
				good.append(m)
		if len(good)>10:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()
		else:
			print ("Not enough matches are found - %d/%d" % (len(good),10))
			matchesMask = None
		draw_params = dict(matchColor = (0,255,0),singlePointColor = None,matchesMask = matchesMask,flags = 2)
		if ret:
			p1 = (int(bbox[0]), int(bbox[1]))
			p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0))
		if ret1:
			p1 = (int(bbox1[0]), int(bbox1[1]))
			p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0))
		if ret2:
			p1 = (int(bbox2[0]), int(bbox2[1]))
			p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0))
		img3 = cv2.drawMatches(refimage,kp1,frame,kp2,good,None,**draw_params)
		r = 1000 / img3.shape[1]
		dim = (1000, int(img3.shape[0] * r))
		resized = cv2.resize(img3, dim, interpolation = cv2.INTER_AREA)
		fps = cap.get(cv2.CAP_PROP_FPS)
		print ("Frames per second are : {0}".format(fps))
		cv2.imshow('frame',resized)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()