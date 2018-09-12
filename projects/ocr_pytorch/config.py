# -*- coding: utf-8 -*-
""" config for characterOCR
"""
import argparse

with open('wordset/words.txt', encoding='utf-8') as fwords:
    words = fwords.read().strip()

WORDSET = words


def arg_parse():
    parser=argparse.ArgumentParser(description='Pytorch OCR')
    parser.add_argument("--wordset", dest="wordset", default=WORDSET)
    parser.add_argument("--wordset_size", dest="wordset_size", default=len(WORDSET))
    parser.add_argument("--image_size", dest="image_size", default=64)
    parser.add_argument("--batch_size", dest='batch_size', default=128)
    parser.add_argument("--epochs", dest="epochs", default=20)
    parser.add_argument("--train_data_dir", dest='train_data_dir', default="D:/Mory/ocr_sina/train/original")
    parser.add_argument("--test_data_dir", dest='test_data_dir', default='D:/Mory/ocr_sina/train/test')
    parser.add_argument("--restore", dest='restore', default=False)
    parser.add_argument("--eval_steps", dest="eval_steps", default=100)
    parser.add_argument("--save_steps", dest="save_steps", default=1000)
    parser.add_argument("--model", dest='model', default='best.pth')
    return parser.parse_args()

args = arg_parse()