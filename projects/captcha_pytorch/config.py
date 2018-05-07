import argparse
from string import ascii_letters, digits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', help="batch size of model training", type=int, default=64)
    parser.add_argument('train_epoch', help="training epoch", type=int, default=3000)
    parser.add_argument('image_width', help='width of training image', type=int)
    parser.add_argument('image_height', help='height of training image', type=int)
    parser.add_argument('train_data_dir', help='directory of train data set', type=str)
    parser.add_argument('test_data_dir', help='directory of testing', type=str)
    args = parser.parse_args()
    args.charset = ascii_letters + digits
    return args