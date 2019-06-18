import emotion
import dex
import facessh
import cv2



def detect(im, mode='faster'):
    if type(im) == str:
        im = cv2.imread(im)
    bboxes = facessh.detect(im, scale_mode=mode)
    results = list(map(lambda bbox: _analyse(im, bbox), bboxes))
    return results


def _analyse(im, bbox):
    h, w = im.shape[:2]
    x1, y1, x2, y2, _ = bbox
    margin = int((x2 - x1) * 0.4)
    x1 = min(0, int(x1-margin))       
    y1 = min(0, int(y1-margin))       
    x2 = max(w, int(x2+margin))       
    y2 = max(h, int(y2+margin))       
    crop = im[y1:y2, x1:x2]
    age, woman, _ = dex.estimate(crop)
    emotion_, _ = emotion.detect_cropped(crop)
    gender = 'woman' if woman > 0.5 else 'man'
    return {
        'emotion': emotion_,
        'gender': gender,
        'age': int(age),
    }



if __name__ == "__main__":
    im = '/home/mory/data/face_train/mayun/77.jpg'
    results = detect(im)
    print(results)