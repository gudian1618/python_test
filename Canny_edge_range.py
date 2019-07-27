import cv2
import numpy as np


def nothing(x):
	# edges = cv2.Canny(img,0,255,True)
	pass


img = cv2.imread('b.jpg')
cv2.namedWindow('image')

cv2.createTrackbar('min', 'image', 0, 255, nothing)
cv2.createTrackbar('max', 'image', 0, 255, nothing)

# edges = cv2.Canny(img,0,255)

while (1):
	
	min1 = cv2.getTrackbarPos('min', 'image')
	max1 = cv2.getTrackbarPos('max', 'image')
	edges = cv2.Canny(img, min1, max1, True)
	cv2.imshow('image', edges)
	
	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()