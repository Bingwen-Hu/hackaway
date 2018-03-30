import cv2

model = cv2.CascadeClassifier('./face_model.xml')
filename = '13.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.imwrite(f"./my{filename}", img)