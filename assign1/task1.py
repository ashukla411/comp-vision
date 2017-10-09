import numpy as np
import cv2

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=-5)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=-5)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad
# cv2.imshow('as',img_color)
# cv2.waitKey(0) 
im =  cv2.imread("B:/UNSW courses/9517 computer vision/ass1/DataSamples/test_sample2.jpg", cv2.IMREAD_GRAYSCALE);
sz = im.shape
print sz
height = int(sz[0] / 3);
width = sz[1]
im_color = np.zeros((height,width,3), dtype=np.uint8 )
for i in xrange(0,3) :
	im_color[:,:,i] = im[ i * height:(i+1) * height,:]
im_aligned = np.zeros((height,width,3), dtype=np.uint8 )
im_aligned[:,:,2] = im_color[:,:,2]
warp_mode = cv2.MOTION_TRANSLATION
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-10)
if height<700:
	for i in xrange(0,2) :
		(cc, warp_matrix) = cv2.findTransformECC (get_gradient(im_color[:,:,2]), get_gradient(im_color[:,:,i]),warp_matrix, warp_mode, criteria)
		im_aligned[:,:,i] = cv2.warpAffine(im_color[:,:,i], warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	# Show final output
	# cv2.imshow("Color Image", im_color)
	# cv2.imshow("Aligned Image", im_aligned)
	cv2.imwrite('result1.jpg',im_aligned)
	cv2.waitKey(0)
else:
	levels=int(height/700);
	i=0;
	dst=im
	while (i<=levels):
		dst=cv2.pyrDown(dst)
		i=i+1
	sz = dst.shape
	print sz
	height = int(sz[0] / 3);
	width = sz[1]
	im_color = np.zeros((height,width,3), dtype=np.uint8 )
	for i in xrange(0,3) :
		im_color[:,:,i] = dst[ i * height:(i+1) * height,:]
	im_aligned = np.zeros((height,width,3), dtype=np.uint8 )
	im_aligned[:,:,2] = im_color[:,:,2]
	warp_mode = cv2.MOTION_TRANSLATION
	warp_matrix = np.eye(2, 3, dtype=np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000,  1e-10)
	for i in xrange(0,2) :
		(cc, warp_matrix) = cv2.findTransformECC (get_gradient(im_color[:,:,2]), get_gradient(im_color[:,:,i]),warp_matrix, warp_mode, criteria)
		im_aligned[:,:,i] = cv2.warpAffine(im_color[:,:,i], warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	# Show final output
	# cv2.imshow("Color Image", im_color)
#	cv2.imshow("Pyramid Down Aligned", im_aligned)
	res = cv2.resize(im_aligned,None,fx=levels+1, fy=levels+1, interpolation = cv2.INTER_CUBIC)
#	cv2.imshow("Pyramid Up Aligned",res)
	cv2.imwrite('result2.jpg',res)
	cv2.waitKey(0)