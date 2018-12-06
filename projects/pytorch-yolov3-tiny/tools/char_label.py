# tools script with labeling.exe tool
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from uuid import uuid1
from PIL import Image


def char_crop(image_id):
    img = Image.open(f"img/{image_id}.png")
    in_file = open(f'ann/{image_id}.xml')
    
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        box = [
            float(xmlbox.find('xmin').text), 
            float(xmlbox.find('xmax').text), 
            float(xmlbox.find('ymin').text), 
            float(xmlbox.find('ymax').text),
        ]
        char = img.crop([box[0], box[2], box[1], box[3]])
        try:
            char.save(f"crop/{uuid1()}.png")
        except:
            print(image_id)

    in_file.close()
    

if __name__ == '__main__':
    source_dir = "/home/jenny/datasets/wechat"
    os.chdir(source_dir)
    imagefiles = os.listdir('ann')
    for imagefile in imagefiles:
        image_id = imagefile.split('.')[0]
        char_crop(image_id)
    
