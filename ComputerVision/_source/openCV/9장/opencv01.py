# -*-coding: utf-8 -*-
import cv2

cam = cv2.VideoCapture(cv2.CAP_DSHOW, 0)
while (cam.isOpened()):
    ret, frame = cam.read()
    cv2.imshow("press 'q'", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
