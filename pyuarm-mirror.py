#!/usr/bin/env python

import numpy as np
import cv2
import pyuarm

arm = pyuarm.uarm.UArm()

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

#Masking face out so it doesn't get detected along with hand
#From opencv-3.1.0/samples/python/facedetect.py
#def detect(img, cascade):
#    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
#                                     flags=cv2.CASCADE_SCALE_IMAGE)
#    if len(rects) == 0:
#        return []
#    rects[:,2:] += rects[:,:2]
#    return rects

#def draw_rects(img, rects, color):
#    for x1, y1, x2, y2 in rects:
#        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

#Allow HSV masking to be changed on the fly to adapt to different environments	
cv2.createTrackbar("Lower Hue", "Mask", 0, 180, updateMask)
cv2.createTrackbar("Upper Hue", "Mask", 0, 180, updateMask)
cv2.createTrackbar("Lower Saturation", "Mask", 0, 255, updateMask)
cv2.createTrackbar("Upper Saturation", "Mask", 0, 255, updateMask)
cv2.createTrackbar("Lower Value", "Mask", 0, 255, updateMask)
cv2.createTrackbar("Upper Value", "Mask", 0, 255, updateMask)

#Good default values
cv2.setTrackbarPos("Lower Hue", "Mask", 0)
cv2.setTrackbarPos("Upper Hue", "Mask", 34)
cv2.setTrackbarPos("Lower Saturation", "Mask", 36)
cv2.setTrackbarPos("Upper Saturation", "Mask", 255)
cv2.setTrackbarPos("Lower Value", "Mask", 0)
cv2.setTrackbarPos("Upper Value", "Mask", 255)

cascade_fn = "model/haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_fn)

while True:
	ret, frame = cam.read()

	if ret:
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(frame_hsv, lower, upper)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
		mask = cv2.dilate(mask, kernel, iterations=3)
		mask = cv2.erode(mask, kernel, iterations=5)
		mask = cv2.GaussianBlur(mask, (5, 5), 0)

		#face_mask = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2GRAY)
		#face_mask = cv2.equalizeHist(face_mask)
		#rects = detect(face_mask, cascade)
		#draw_rects(frame, rects, (0,0,0))

		_, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		#Find largest contour (this assumes it's the hand)
		if len(contours) > 0:
			maxArea = 0
			contour = 0
			for i in range(len(contours)):
				area = cv2.contourArea(contours[i])
				if area > maxArea:
					area = maxArea
					contour = i
			
			hand = contours[contour]
			hull = cv2.convexHull(hand)
			hullPts = cv2.convexHull(hand, returnPoints=False)
			defects = cv2.convexityDefects(hand, hullPts)

			pump = False
			if defects is not None:
				pump = True
				for defect in defects:
					if defect[0][3] > 10000:
						pump = False
						break

			#print pump
			#arm.set_pump(pump)

			M = cv2.moments(hand)

			if int(M["m00"]) is not 0:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])

				cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)

				arm_y = 300 - (0.235849 * cX + 100)
				arm_z = (-0.3125 * cY + 250)
				arm.set_position(0, arm_y, arm_z, 100)

			cv2.drawContours(frame, contours, contour, (255, 0, 0), 3)
			cv2.drawContours(frame, [hull], 0, (0, 0, 255), 3)

		cv2.imshow("Camera", np.hstack((frame, cv2.bitwise_and(frame, frame, mask=mask))))
		k = cv2.waitKey(1) & 0xFF
		if k == 27:
			break
	else:
		print "Could not read frame"
		break

arm.set_pump(False)

cam.release()
cv2.destroyAllWindows()

arm.disconnect()

