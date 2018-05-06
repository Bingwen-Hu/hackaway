import argparse
from string import ascii_lowercase, digits, ascii_uppercase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help="batch size of model training", type=int, default=64)
    parser.add_argument('--epoch', help="training epoch", type=int, default=3000)
    parser.add_argument('--image_width', help='width of training image', type=int, default=100)
    parser.add_argument('--image_height', help='height of training image', type=int, default=40)
    parser.add_argument('--train_data_dir', help='directory of train data set', type=str, default='E:/captcha-data/dwnews/train')
    parser.add_argument('--test_data_dir', help='directory of testing', type=str, default='E:/captcha-data/dwnews/test')
    parser.add_argument('--captcha_size', help='number of captcha character', type=int, default=4)
    args = parser.parse_args()
    args.charset = ascii_lowercase
    return args