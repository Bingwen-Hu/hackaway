# tensorflow image handle
import matplotlib.pyplot as plt
import tensorflow as tf

image = plt.imread("E:/tmp/soitis.jpg")
height, width, depth = image.shape
# setting up tf.Session
sess = tf.Session()
sess.as_default()


# convert image type
image_g = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
image_r = sess.run(image_g)
print("type: %r" % type(image_r))


# resize, pad and crop
image_r = tf.image.resize_image_with_crop_or_pad(image, height+8, width+8)
image_r2 = sess.run(image_r)
print('type: %r' % type(image_r2))

image_c = tf.random_crop(image_r, [height, width, depth])
image_c2 = sess.run(image_c)
print('type: %r' % type(image_c2))

# brightness, contrast, saturation, hue
image_b = tf.image.random_brightness(image, max_delta=0.1)
image_c = tf.image.random_contrast(image_b, lower=0.5, upper=1)
image_s = tf.image.random_saturation(image_c, lower=0.1, upper=1)
image_h = tf.image.random_hue(image_s, max_delta=0.5)
image_b2 = sess.run(image_h)
print('type: %r' % type(image_b2))