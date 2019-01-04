import cv2
import time

cap = cv2.VideoCapture("/home/mory/Downloads/org.avi")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("frame", frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.040)

cap.release()
cv2.destroyAllWindows()