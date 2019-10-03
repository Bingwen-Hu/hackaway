# data preprocess
# we want to mask out the uppercloth region and arms
import os
import os.path as osp
import numpy as np
from PIL import Image

# data directory
class Option:
    mode = 'test'
    person_dir = f'data/{mode}/image'
    parse_dir = f'data/{mode}/image-parse'
    mask_dir = f'data/{mode}/image-mask'


# parse region order
parse_regions = [
    'Background',  # always index 0
    'Hat', 'Hair', 'Glove', 'Sunglasses',
    'UpperClothes', 'Dress', 'Coat', 'Socks',
    'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
    'Face', 'Left-arm', 'Right-arm', 'Left-leg',
    'Right-leg', 'Left-shoe', 'Right-shoe',
]

mask_regions = [
    'UpperClothes', 'Coat', 
    'Left-arm', 'Right-arm', 
]

mask_indice = [parse_regions.index(r) for r in mask_regions]


def mask_person(opt, person_id, mask_indice, mask_value):
    person_path = osp.join(opt.person_dir, person_id)
    parse_path = osp.join(opt.parse_dir, person_id.replace('.jpg', '.png'))

    person_im = Image.open(person_path)
    parse_im = Image.open(parse_path)
    
    person_array = np.array(person_im)
    parse_array = np.array(parse_im)

    mask = np.zeros_like(parse_array, dtype=np.bool)
    for index in mask_indice:
        mask = mask | (parse_array == index)
    person_array[mask, :3] = mask_value
    mask_path = osp.join(opt.mask_dir, person_id.replace('.jpg', '.png'))
    Image.fromarray(person_array).save(mask_path)

    return person_array

def draw_neck(opt, person_id, mask_value):
    # TODO: draw a rectangle at neck region
    pass


if __name__ == '__main__':

    opt = Option()
    os.makedirs(opt.mask_dir, exist_ok=True)
    for i, person_id in enumerate(os.listdir(opt.person_dir)):
        mask_person(opt, person_id, mask_indice, mask_value=128)
        print(f"Generate Mask Person: {i:5d}, {person_id}")

