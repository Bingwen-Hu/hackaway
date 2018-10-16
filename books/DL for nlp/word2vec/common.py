import random
import collections
import math
import os
import zipfile
import time
import re
import numpy as np
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve

dataset_link = 'http://mattmahoney.net/dc/'
zip_file = 'text8.zip'

def data_download(zip_file):
    """Downloading the required file"""
    if not os.path.exists(zip_file):
        zip_file, _ = urlretrieve(dataset_link + zip_file, zip_file)
        print('File download successfully!')
    return None

data_download(zip_file)

extracted_folder = 'dataset'

if not os.path.isdir(extracted_folder):
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(extracted_folder)

with open('dataset/text8') as ft_:
    full_text = ft_.read()

def text_processing(ft8_text):
    ft8_text = ft8_text.lower()
    ft8_text = ft8_text.replace('.', ' <period> ')
    ft8_text = ft8_text.replace(',', ' <comma> ')
    ft8_text = ft8_text.replace('"', ' <quotation> ')
    ft8_text = ft8_text.replace(';', ' <semicolon> ')
    ft8_text = ft8_text.replace('!', ' <exclamation> ')
    ft8_text = ft8_text.replace('?', ' <question> ')
    ft8_text = ft8_text.replace('(', ' <paren_l> ')
    ft8_text = ft8_text.replace(')', ' <paren_r> ')
    ft8_text = ft8_text.replace('--', ' <hyphen> ')
    ft8_text = ft8_text.replace(':', ' <colon> ')
    ft8_text_token = ft8_text.split()
    return ft8_text_token
    
ft_token = text_processing(full_text)

word_cnt = collections.Counter(ft_token)
shortlisted_words = [w for w in ft_token if word_cnt[w] > 7]
print(shortlisted_words[:15])
print('Total number of shortlisted words: ', len(shortlisted_words))
print('Unique number of shortlisted words: ', len(set(shortlisted_words)))

def dict_creation(shortlisted_words):
    """
    here, one word map into one int, frequency is disgarded
    """
    counts = collections.Counter(shortlisted_words)
    vocabulary = sorted(counts, key=counts.get, reverse=True)
    rev_dictionary_ = {ii: word for ii, word in enumerate(vocabulary)}
    dictionary_ = {word: ii for ii, word in rev_dictionary_.items()}
    return dictionary_, rev_dictionary_

dictionary_, rev_dictionary_ = dict_creation(shortlisted_words)
words_cnt = [dictionary_[word] for word in shortlisted_words]