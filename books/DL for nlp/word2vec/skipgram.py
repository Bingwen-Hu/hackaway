# A subsampling approach has been coupled in the skip-gram model to deal
# with the stopwords in the text. All the words with higher frequency and
# without any significant context around the center words are removed by
# putting a threshold on their frequency. This results in faster training and
# better word vector representations.
import os
import collections
import random
import time

import numpy as np
import tensorflow as tf

from common import words_cnt, rev_dictionary_
### subsampling 
thresh = 0.00005
word_counts = collections.Counter(words_cnt)
total_count = len(words_cnt)
freqs = {word: count / total_count for word, count in word_counts.items()}
p_drops = {word: 1 - np.sqrt(thresh/freqs[word]) for word in word_counts}
train_words = [word for word in words_cnt if p_drops[word] < random.random()]


def skipG_target_set_generation(batch_, batch_index, word_window):
    """return the surround words of the word at batch_index"""
    random_num = np.random.randint(1, word_window+1)
    words_start = batch_index - random_num if (batch_index - random_num) > 0 else 0
    words_stop = batch_index + random_num
    window_target = set(batch_[words_start:batch_index]  + batch_[batch_index+1:words_stop+1])
    return list(window_target)

def skipG_batch_creation(short_words, batch_length, word_window):
    batch_cnt = len(short_words) // batch_length
    short_words = short_words[:batch_cnt*batch_length]

    for word_index in range(0, len(short_words), batch_length):
        input_words, label_words = [], []
        word_batch = short_words[word_index:word_index+batch_length]
        for index_ in range(len(word_batch)):
            batch_input = word_batch[index_]
            batch_label = skipG_target_set_generation(word_batch, index_, word_window)
            # set the input and target same size
            label_words.extend(batch_label)
            input_words.extend([batch_input] * len(batch_label))
            yield input_words, label_words




tf_graph = tf.Graph()
with tf_graph.as_default():
    input_ = tf.placeholder(tf.int32, [None], name='input_')
    label_ = tf.placeholder(tf.int32, [None, None], name='label_')

vocabulary_size = len(rev_dictionary_)
with tf_graph.as_default():
    word_embed = tf.Variable(tf.random_uniform(shape=(vocabulary_size, 300), minval=-1, maxval=1))
    embedding = tf.nn.embedding_lookup(word_embed, input_)

# The code includes the following :
# Initializing weights and bias to be used in the softmax layer
# Loss function calculation using the Negative Sampling
# Usage of Adam Optimizer
# Negative sampling on 100 words, to be included in the loss function
# 300 is the word embedding vector size

with tf_graph.as_default():
    sf_weights = tf.Variable(tf.truncated_normal(shape=(vocabulary_size, 300), stddev=0.1))
    sf_bias = tf.Variable(tf.zeros(vocabulary_size))

    loss_fn = tf.nn.sampled_softmax_loss(weights=sf_weights, biases=sf_bias, 
        labels=label_, inputs=embedding, num_sampled=100, num_classes=vocabulary_size)
    cost_fn = tf.reduce_mean(loss_fn)
    optim = tf.train.AdamOptimizer().minimize(cost_fn)


# The below code performs the following operations :
# Performing validation here by making use of a random
# selection of 16 words from the dictionary of desired size
# Selecting 8 words randomly from range of 1000
# Using the cosine distance to calculate the similarity
# between the words

with tf_graph.as_default():
    validation_cnt = 16
    validation_dict = 100

    words1 = np.array(random.sample(range(validation_dict), validation_cnt//2))
    words2 = np.array(random.sample(range(1000, 1000+validation_dict), validation_cnt//2))
    validation_words = np.append(words1, words2)
    validation_data = tf.constant(validation_words, dtype=tf.int32)

    normalization_embed = word_embed / (tf.sqrt(tf.reduce_sum(tf.square(word_embed), 1, keep_dims=True)))
    validation_embed = tf.nn.embedding_lookup(normalization_embed, validation_data)
    word_similarity = tf.matmul(validation_embed, tf.transpose(normalization_embed))
    


# checkpoint 
if not os.path.exists('model_checkpoint'):
    os.mkdir('model_checkpoint')

epochs = 2
batch_length = 1000
word_window = 10

with tf_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=tf_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs+1):
        batches = skipG_batch_creation(train_words, batch_length, word_window)
        start = time.time()
        for x, y in batches:
            train_loss, _ = sess.run([cost_fn, optim], 
                                    feed_dict={input_: x, label_: np.array(y)[:, None]})
            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print(f'Epoch {e}/{epochs}, Iteration: {iteration}, '
                      f'Avg. Training loss: {loss/100:.4f}, '
                      f'Processing: {(end-start)/100} sec/batch')
                loss = 0
                start = time.time()
            if iteration % 2000 == 0:
                similarity_ = word_similarity.eval()
                for i in range(validation_cnt):
                    validation_words = rev_dictionary_[validation_words[i]]
                    top_k = 8
                    nearest = (-similarity_[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % validation_words
                    for k in range(top_k):
                        close_word = rev_dictionary_[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)
            iteration += 1
    save_path = saver.save(sess, 'model_checkpoint/skipGram_text8.ckpt')
    embed_mat = sess.run(normalization_embed)