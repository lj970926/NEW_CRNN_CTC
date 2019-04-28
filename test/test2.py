import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
BATCH_DECAY = 0.999

def get_feature_map(X):
    '''

    :param X: the input tensor
    :return: a feature map that has two dimension
    '''
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        activation_fn=tf.nn.relu):
        net = slim.repeat(X, 2, slim.conv2d, 64, kernel_size=3, stride=1, scope='conv1')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, kernel_size=3, stride=1, scope='conv2')
        net = slim.max_pool2d(net, kernel_size=2, stride=2, scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv3')
        net = slim.repeat(net, 2, slim.conv2d, 256, kernel_size=3, stride=1, scope='conv4')
        net = slim.max_pool2d(net, [2, 1], stride=[2, 1], scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 512, kernel_size=3, stride=1, scope='conv5')
        net = slim.batch_norm(net, decay=BATCH_DECAY, is_training=True, scope='bn1')
        net = slim.conv2d(net, 512, kernel_size=3, stride=1, scope='conv6')
        net = slim.batch_norm(net, decay=BATCH_DECAY, is_training=True, scope='bn2')
        net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], scope='pool5')
        net = slim.conv2d(net, 512, padding="VALID", kernel_size=[2, 1], stride=1, scope='conv7')
    return net


x = np.random.rand(3,32,100,1)
x = x.astype('float32')
net = get_feature_map(x)
print(net.get_shape().as_list())
