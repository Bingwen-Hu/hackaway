import emotion
import dex
import facessh
import cv2
import pcn


def detect(im, mode='fast'):
    if type(im) == str:
        im = cv2.imread(im)
    bboxes = facessh.detect(im, scale_mode=mode)
    results = list(map(lambda bbox: _analyse(im, bbox), bboxes))
    return results


def _analyse(im, bbox):
    h, w = im.shape[:2]
    x1, y1, x2, y2, _ = bbox
    margin = int((x2 - x1)*0.5)
    x1_ = max(0, int(x1-margin))       
    y1_ = max(0, int(y1-margin))       
    x2_ = min(w, int(x2+margin))       
    y2_ = min(h, int(y2+margin))       
    crop = im[y1_:y2_, x1_:x2_]
    winlist = pcn.detect(crop)
    if len(winlist) != 0:
        crop_pcn, _ = pcn.crop(crop, winlist)[0] # ignore points
    else:
        crop_pcn = crop
    age, woman, _ = dex.estimate(crop_pcn)
    emotion_, _ = emotion.detect_cropped(crop_pcn)
    gender = 'woman' if woman > 0.5 else 'man'
    return {
        'emotion': emotion_,
        'gender': gender,
        'age': int(age),
        'bbox': [x1, y1, x2, y2],
    }


if __name__ == "__main__":
    im = '/home/mory/data/face_train/mayun/77.jpg'
    results = detect(im)
    print(results)