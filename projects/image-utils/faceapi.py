# auto label image using face++ API
import requests
import glob
import os
from uuid import uuid1

from PIL import Image
from utils import padding


api_key = '7zuC2S1YbMWr4Tcs7l1Igk6QB41mpQwj'
api_secret = 'TOP4TM1X-ITyjXTQBbzGHB0ByG_StWBc'

def char_detect(api_key, api_secret, image):
    url = f'https://api-cn.faceplusplus.com/imagepp/v1/recognizetext?api_key={api_key}&api_secret={api_secret}'
    files = {"image_file": (image, open(image, 'rb'), 'image/jpg')}
    r = requests.post(url, files=files)
    return r.json()




imagedirs = glob.glob("/home/mory/Pictures/crop/*.jpg") 
for imgpath in imagedirs:
    img = Image.open(imgpath)
    img = padding(img, (100, 100), True)
    img.save('temp.jpg')
    res = char_detect(api_key, api_secret, 'temp.jpg')
    break


def crop_result(res):
    result = res['result']
    textlines = [r['child-objects'] for r in result if r['type'] == 'textline']
    charlist = [d for lst in textlines for d in lst]
