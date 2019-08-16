import sys
import cv2
import emotion

img = cv2.imread(sys.argv[1])
results = emotion.detect(img)

mapping = {
    'neutral': '平静',
    'happy': '开心',
    'sad': '伤心',
    'disgust': '厌恶',
    'fear': '恐惧',
    'surprise': '惊讶',
    'angry': '生气',
}


LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}


for result in results:
    x1, y1, x2, y2 = result['bbox']
    print(x1,y1,x2,y2)
    emotion = result['emotion']
    emotion = mapping[emotion]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.putText(img, emotion, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

cv2.imshow("Emotion", img)
cv2.imwrite("Emotion.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()