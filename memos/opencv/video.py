import numpy as np
import cv2

cap = cv2.VideoCapture("E:/Mory/0.mp4")
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('E:/Mory/output.avi', fourcc, 20.0, (240, 120))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


# ffmpeg or gstreamer must be installed
# something wrong!
