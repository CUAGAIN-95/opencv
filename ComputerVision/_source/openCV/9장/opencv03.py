# grayscale video

import cv2

cam = cv2.VideoCapture(cv2.CAP_DSHOW, 0)

while (cam.isOpened()):
    ret, frame = cam.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray_frame)
    cv2.imshow("origin", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
