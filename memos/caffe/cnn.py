import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()


net = caffe.Net('conv.prototxt', caffe.TEST)

# network arch
print(net.blobs['data'])
for k, v in net.blobs.items():
    print(k, v.data.shape)

# params
print("weights: ", net.params['conv'][0].data.shape)
print("bias: ", net.params['conv'][1].data.shape)


im = np.array(Image.open('cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

# compute
net.forward()


net.save('cnn.caffemodel')
