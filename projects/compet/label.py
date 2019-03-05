def convert(size, box):
    """
    size: image size
    bbox: left-top (x, y) and (width, height) 
    """
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0] + box[2] / 2 
    y = box[1] + box[3] / 2
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]



def label(images, annotations):
    #result = images.merge(annotations, left_on='id', right_on='image_id') 
    #result = result.drop(['coco_url', 'data_captured', 'flickr_url', 'license', 'iscrowd', 'segmentation'], axis=1)
    for id_ in images.id:
        ann = annotations[annotations.image_id == id_]
        bbox = ann.bbox.tolist()
        classes = ann.category_id.tolist()
        img = images[images.id == id_]
        img_name = img.file_name.values[0]
        size = img.width.values[0], img.height.values[0]
        with open(f'data/round1_train/restricted_ann/{img_name[:-4]}.txt', 'w') as f:
            for class_, bbox_ in zip(classes, bbox):
                bbox_ = convert(size, bbox_)
                bbox_.insert(0, class_)
                bbox_ = [str(b) for b in bbox_]
                f.write(' '.join(bbox_)+'\n')

