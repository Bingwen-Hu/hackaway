import numpy as np
import cv2


cap = cv2.VideoCapture(0)

if cap.isOpened():
    print('good!')
else:
    cap.open(0)
    print('whether open? {}'.format(cap.isOpened()))

while(True):
    # capture frame by frame
    ret, frame = cap.read()
    
    # our operations on the frame
    cv2.imshow('camera', frame)
    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
