# from statistics import mode


# import cv2
# from keras.models import load_model
# import numpy as np
# import landmark


# def preprocess_input(x, v2=True):
#     x = x.astype('float32')
#     x = x / 255.0
#     if v2:
#         x = x - 0.5
#         x = x * 2.0
#     return x

# # parameters for loading data and images
# emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
# emotion_classifier = load_model(emotion_model_path, compile=False)
# emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}



# img = cv2.imread('test.jpg')
# results = landmark.detect('test.jpg')
# result = results[0]
# x1, y1, x2, y2 = result['bbox']
# face = img[y1:y2, x1:x2, :]
# img = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (64, 64))
# img = preprocess_input(img)
# img = img[None, :, :, None]
# res = emotion_classifier.predict(img)

import cv2
import emotion

img = cv2.imread('test.jpg')
results = emotion.detect(img)

for result in results:
    x1, y1, x2, y2 = result['bbox']
    emotion = result['emotion']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

cv2.imshow("Emotion", img)
cv2.waitKey(0)
cv2.destroyAllWindows()