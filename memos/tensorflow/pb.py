import tensorflow as tf
from tensorflow.python.platform import gfile

model = '/home/mory/URunResearchPrototypeCode/quickdemo/IntellVideo/face-recognize/model/20180402-114759.pb'



 
sess = tf.Session()
with gfile.FastGFile(model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='face') # 导入计算图
 
# 需要有一个初始化的过程    
sess.run(tf.global_variables_initializer())
 

def compute_emb(sess, images):
    # Get input and output tensors
    images_placeholder = sess.get_tensor_by_name("input:0")
    embeddings = sess.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = sess.get_tensor_by_name("phase_train:0")
    # Run forward pass to calculate embeddings
    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb

