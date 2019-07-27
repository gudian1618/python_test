import cv2


# import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while 1:
	ret, frame = cap.read()
	cv2.imshow('cap', frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()