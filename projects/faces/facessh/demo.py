import cv2
import facessh


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    results = facessh.detect(frame, scale_mode='fastest')
    im = facessh.draw(frame, results)
    cv2.imshow("ssh", im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()