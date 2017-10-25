""" 使用inception计算图片特征，比对1008个值的前10，相似度度量分为两步：
1、比对索引的相似度，索引值相等的个数越多，图片越相似
2、比对索引值的差值，差值越小，图片越相似
"""

import os
from glob import glob
import numpy as np
import tensorflow as tf



flags = tf.app.flags
flags.DEFINE_string("model_dir", "model", "model directory")
flags.DEFINE_string("model_file", "classify_image_graph_def.pb", "model file name")
flags.DEFINE_string('input_tensor_name', 'DecodeJpeg/contents:0', 'name of input tensor')
flags.DEFINE_string('output_tensor_name', 'softmax:0', 'name of output tensor')
flags.DEFINE_string('cache_dir', '/tmp/videosearch/', "cache directory")
flags.DEFINE_string('input_dir', '', "directory that collect")
flags.DEFINE_string('image_file', '', 'input image file')
flags.DEFINE_integer('k', 10, 'top k')
flags.DEFINE_string('mode', '', "`cache` or `match` or `compute`")
FLAGS = flags.FLAGS


def create_graph_helper():
    """创建计算图，加载模型文件"""
    with tf.gfile.FastGFile(os.path.join(
	    FLAGS.model_dir, FLAGS.model_file), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")


def get_image_top_k(sess, imagepath):
    """CNN计算并返回k个最佳匹配的索引和概率值
    sess : tensorflow.Session
    image: 输入图片的路径
    k    : top_k中的k值，表示前几
    ret  : top_k, probs 索引和概率值
    """
    if not tf.gfile.Exists(imagepath):
        tf.logging.fatal('File does not exist %s', imagepath)
    image_data = tf.gfile.FastGFile(imagepath, 'rb').read()

    softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.output_tensor_name)
    predictions = sess.run(softmax_tensor,
                           {FLAGS.input_tensor_name: image_data})
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-FLAGS.k:][::-1]
    probs = [predictions[t] for t in top_k]
    return top_k, probs


def write_image_top_k(cache_dir, imagepath, top_k, probs):
    """ 缓存单个图片的特征值，将top_k与probs用·|·隔开
    cache_dir: 缓存的目录
    imagepath: 图片的路径
    top_K    : top_k的索引值
    probs    : top_k对应的概率值
    """
    top_k_string = ','.join(str(x) for x in top_k)
    probs_string = ','.join(str(x) for x in probs)
    final_string = top_k_string + '|' + probs_string
    basename = os.path.basename(imagepath)
    cachepath = os.path.join(cache_dir, basename)
    if not os.path.exists(cachepath):
        with open(cachepath, "w") as f:
            f.write(final_string)


def cache_data(cache_dir, input_dir):
    """将输入图片计算一个分类，然后缓存起来
    cache_dir: 缓存目录
    input_dir: 图片输入目录
    """
    image_paths = glob(os.path.join(input_dir, "*.jpg"))
    create_graph_helper()

    with tf.Session() as sess:
        for path in image_paths:
            print("cache {}...".format(os.path.basename(path)))
            top_k, probs = get_image_top_k(sess, path)
            write_image_top_k(cache_dir, path, top_k, probs)


def read_cache_data(cache_dir):
    """读取缓存文件，以文件名键，匹配的索引及概率为值"""
    cache_files = glob(os.path.join(cache_dir, '*'))
    cache_dict = {}
    for file in cache_files:
        with open(file) as f:
            top_k, probs = f.read().split('|')
            tup = (np.fromstring(top_k, sep=','), np.fromstring(probs, sep=','))
            basename = os.path.basename(file)
            cache_dict.update({basename:tup})
    return cache_dict


def find_most_similar(top_k, probs, cache_dict, num=10):
    """返回最相似的num张照片的文件名，如果找到相似的，
    则返回一个包括匹配元组的列表，否则返回一个空列表
    top_k     : 包含最佳分类的索引的列表
    probs     : 包含最佳分类索引对应的概率
    cache_dict: 缓存中的索引和概率
    num       : 返回最近匹配的数目
    """
    similar = []
    for filename in cache_dict:
        score = 0
        count = 0
        other_top_k, other_probs = cache_dict[filename]
        for i, t in enumerate(top_k):
            if t in other_top_k:
                prob = probs[i]
                other_prob = other_probs[other_top_k.tolist().index(t)]
                score += abs(prob-other_prob)
                count += 1
        if count > 0:
            score = score / count
            similar.append((filename, score))
    if similar:
        similar.sort(key=lambda item: item[1]) # 根据score升序排序
        return similar[:num]
    return similar

def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "cache":
        if not os.path.exists(FLAGS.cache_dir):
            os.makedirs(FLAGS.cache_dir)
        if not os.path.exists(FLAGS.input_dir):
            tf.logging.fatal("input direcotory does not exist!")
        print("cache data into {}...".format(FLAGS.cache_dir))
        cache_data(FLAGS.cache_dir, FLAGS.input_dir)

    elif FLAGS.mode == 'match':
        if not os.path.exists(FLAGS.image_file):
            tf.logging.fatal("{} does not exist!".format(FLAGS.image_file))
        print('read cache from {}...'.format(FLAGS.cache_dir))
        cache_dict = read_cache_data(FLAGS.cache_dir)
        print('compute the features of input image...')
        create_graph_helper()
        with tf.Session() as sess:
            top_k, probs = get_image_top_k(sess, FLAGS.image_file)
            results = find_most_similar(top_k, probs, cache_dict)
        if results:
            for name, score in results:
                print("Match file {}, score {}".format(name, score))
        else:
            print("Could not match any file!")

    elif FLAGS.mode == 'compute':
        tf.logging.info("compute mode just for test")
        create_graph_helper()
        with tf.Session() as sess:
            top_k, probs = get_image_top_k(sess, FLAGS.image_file)
        print('top k is: ', top_k)
        print('probs is: ', probs)
    else:
        print("only `cache`, `match` and `compute` available!")

if __name__ == "__main__":
    tf.app.run()