#!/usr/bin/env python

import numpy as np
import cv2
import pyuarm

#arm = pyuarm.uarm.UArm()

cam = cv2.VideoCapture(0)
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

lower = upper = np.array([0, 0, 0])

#Not the most efficient method
def updateMask(self):
	global lower, upper
	lower = np.array([cv2.getTrackbarPos("Lower Hue", "Mask"), 
					cv2.getTrackbarPos("Lower Saturation", "Mask"),
					cv2.getTrackbarPos("Lower Value", "Mask")])
	upper = np.array([cv2.getTrackbarPos("Upper Hue", "Mask"), 
					cv2.getTrackbarPos("Upper Saturation", "Mask"),
					cv2.getTrackbarPos("Upper Value", "Mask")])

#Allow HSV masking to be changed on the fly to adapt to different environments	
cv2.createTrackbar("Lower Hue", "Mask", 0, 180, updateMask)
cv2.createTrackbar("Upper Hue", "Mask", 0, 180, updateMask)
cv2.createTrackbar("Lower Saturation", "Mask", 0, 255, updateMask)
cv2.createTrackbar("Upper Saturation", "Mask", 0, 255, updateMask)
cv2.createTrackbar("Lower Value", "Mask", 0, 255, updateMask)
cv2.createTrackbar("Upper Value", "Mask", 0, 255, updateMask)

#Good default values
cv2.setTrackbarPos("Lower Hue", "Mask", 79)
cv2.setTrackbarPos("Upper Hue", "Mask", 180)
cv2.setTrackbarPos("Lower Saturation", "Mask", 10)
cv2.setTrackbarPos("Upper Saturation", "Mask", 255)
cv2.setTrackbarPos("Lower Value", "Mask", 0)
cv2.setTrackbarPos("Upper Value", "Mask", 255)

while True:
	ret, frame = cam.read()

	if ret:
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(frame_hsv, lower, upper)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
		mask = cv2.dilate(mask, kernel, iterations=1)
		mask = cv2.erode(mask, kernel, iterations=3)

		cv2.imshow("Camera", cv2.bitwise_and(frame, frame, mask=mask))
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
	else:
		print "Could not read frame"
		break

cam.release()
cv2.destroyAllWindows()

