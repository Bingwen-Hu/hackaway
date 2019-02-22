import tensorflow as tf

from model import build_graph
from config import FLAGS
from preprocess import get_X, int2text


def predict(img):
    size = (FLAGS.image_size, FLAGS.image_size)
    image = get_X(img, size)
    predict_opts = [graph['predicted_val_top_k'], graph['predicted_index_top_k']]
    feed_dict = {graph['images']: [image],
                 graph['keep_prob']: 1.0,
                 graph['is_training']: False}
    predict_val, predict_index = sess.run(predict_opts, feed_dict=feed_dict)
    predict_index = predict_index.flatten()
    text = int2text(predict_index[0], FLAGS.wordset)
    return text



# when import from other modules, load the model
sess = tf.Session()
graph = build_graph(top_k=1)
saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
saver.restore(sess, ckpt)