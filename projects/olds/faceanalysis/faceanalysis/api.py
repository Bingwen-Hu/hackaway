import insight # under the same directory
import emotion
import cv2


def analyse(im):
    """
    Returns:
        empty list or list of dictionary
    """
    if type(im) == str:
        im = cv2.imread(im)
    ga_bbox = insight.estimate(im) 
    if len(ga_bbox) == 0:
        return []
    emotions = [analyse_emotion(im, bbox) for (_, bbox) in ga_bbox]
    results = construct_result(ga_bbox, emotions)
    return results


def analyse_emotion(im, bbox):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = bbox
    margin = 0
    x1_ = max(0, int(x1-margin))       
    y1_ = max(0, int(y1-margin))       
    x2_ = min(w, int(x2+margin))       
    y2_ = min(h, int(y2+margin))       
    crop = im[y1_:y2_, x1_:x2_]
    emotion_, _ = emotion.detect_cropped(crop)
    return emotion_


def construct_result(ga_bbox, emotions):
    results = []
    for ((gender, age), bbox), emotion_ in zip(ga_bbox, emotions):
        gender = 'man' if gender == 1 else 'woman'
        results.append({
            'emotion': emotion_,
            'gender': gender,
            'age': age,
            'bbox': bbox,
        })
    return results

