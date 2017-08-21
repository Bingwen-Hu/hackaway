import tensorflow as tf
import os


w = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(0., name="bias")


def inference(X):
    return tf.matmul(X, w) + b

def loss(X, Y):
    Y_predicted = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))

def inputs():
    weight = [84, 73, 65, 70, 76, 69, 63, 72, 79, 75, 27, 89, 65, 
              57, 59, 69, 60, 79, 75, 82, 59, 67, 85, 55, 63]
    age = [46, 20, 52, 30, 57, 25, 28, 36, 57, 44, 24, 31, 52, 
           23, 60, 48, 34, 51, 50, 34, 46, 23, 37, 40, 30]
    weight_age = [[w, a] for w, a in zip(weight, age)]
    blood_fat_content = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 
                         254, 395, 434, 220, 374, 308, 220, 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat_content)
    
def train(total_loss):
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    print(sess.run(inference([[80., 25.]])))
    print(sess.run(inference([[65., 25.]])))




# model definition code.

# create a saver
saver = tf.train.Saver()
    
with tf.Session() as sess:
    
    # model setup
    tf.global_variables_initializer().run()
    
    X, Y = inputs()
    
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    initial_step = 0
    training_steps = 1000
    
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit("-", 1)[1])

    # actual training loop
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        
        if step % 1000 == 0:
            print("Time to save")
            #saver.save(sess, "my-model", global_step=step)
        
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))
        
            
    evaluate(sess, X, Y)
    
    #saver.save(sess, "my-model", global_step=training_steps)
    
    
    coord.request_stop()
    coord.join(threads)
    sess.close()
	
print(__file__)