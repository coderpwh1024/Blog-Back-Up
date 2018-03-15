#-*- coding: utf-8 -*-
# Author: Lawlite
# Date: 2017/07/26
# Associate Blog: http://lawlite.me/2017/06/24/Tensorflow学习-工具相关/#1、可视化计算图
# License: MIT

'''此文件可视化神经网络的结构
   - 最后执行 tensorboard --logdir=logs/ 命令即可， 在浏览器中localhost:6006查看即可
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print("tensorflow版本：", tf.__version__)

'''加载数据'''
data = input_data.read_data_sets('MNIST_data', one_hot=True)
print("Size of:")
print("\t\t training set:\t\t{}".format(len(data.train.labels)))
print("\t\t test set: \t\t\t{}".format(len(data.test.labels)))
print("\t\t validation set:\t{}".format(len(data.validation.labels)))

'''超参数'''
img_size = 28
img_flatten_size = img_size ** 2
img_shape = (img_size, img_size)
num_classes = 10
learning_rate = 1e-4
n_channels = 1
#------------
batch_size = 128
n_steps = 28
state_size = 256
n_inputs = 28

'''定义添加一层全连接层'''
def add_fully_layer(inputs, input_size, output_size, num_layer, activation=None):
    with tf.name_scope('layer_'+num_layer):
        with tf.name_scope('Weights'):
            W = tf.Variable(initial_value=tf.random_normal(shape=[input_size, output_size]), name='W')
        with tf.name_scope('biases'):
            b = tf.Variable(initial_value=tf.zeros(shape=[1, output_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, W) + b
        if activation is not None:
            outputs = activation(Wx_plus_b)
        else:
            outputs = Wx_plus_b
        return outputs
'''CNN 定义添加一层卷积层，包括pooling(使用maxpooling, size=2)'''
def add_conv_layer(inputs, filter_size, input_channels, output_channels, num_layer, activation=tf.nn.relu):
    with tf.name_scope('conv_layer_'+num_layer):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.truncated_normal(stddev=0.1, shape=[filter_size, filter_size, input_channels, output_channels]), name='W')
        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(0.1, shape=[output_channels]))
        with tf.name_scope('conv2d'):
            conv2d_plus_b = tf.nn.conv2d(inputs, Weights, strides=[1,1,1,1], padding='SAME', name='conv') + b
            activation_conv_outputs = activation(conv2d_plus_b)
        with tf.name_scope('max_pool'):
            max_pool_outputs = tf.nn.max_pool(activation_conv_outputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        return max_pool_outputs    
  
  
'''将卷积层展开'''
def flatten_layer(layer):
    '''
    @param layer: the conv layer
    '''
    layer_shape = layer.get_shape() # 获取形状(layer_shape == [num_images, img_height, img_width, num_channels])
    num_features = layer_shape[1:4].num_elements()  # [1:4] 是最后3个维度，就是展开的长度
    layer_flat = tf.reshape(layer, [-1, num_features])   # 展开
    return layer_flat, num_features

'''RNN 添加一层cell'''
def add_RNN_Cell(inputs):
    with tf.name_scope('RNN_LSTM_Cell'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal(shape=[state_size, num_classes]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.1, shape=[num_classes,]), name='b')
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_size)
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=x, 
                                                     initial_state=init_state)
        logits = tf.matmul(final_state[1], weights) + biases
        return logits
  
  
# ==================================================      
'''placehoder'''
with tf.name_scope('inputs'):
    #x = tf.placeholder(tf.float32, shape=[None, img_flatten_size], name='x')
    #y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
    #x_image = tf.reshape(x, shape = [-1, img_size, img_size, n_channels], name='x_images')
    '''RNN'''
    x = tf.placeholder(tf.float32, shape=[batch_size, n_steps, n_inputs], name='x')
    y = tf.placeholder(tf.float32, shape=[batch_size, num_classes], name='y')
    

'''全连接网络结构'''
#hidden_layer1 = add_fully_layer(x, img_flatten_size, 20, '1', activation=tf.nn.relu)
#logits = add_fully_layer(hidden_layer1, 20, num_classes, '2')


'''CNN卷积网络结构'''
#conv_layer1 = add_conv_layer(x_image, filter_size=5, input_channels=1, 
                            #output_channels=32, 
                            #num_layer='1')
#conv_layer2 = add_conv_layer(conv_layer1, filter_size=5, input_channels=32, 
                            #output_channels=64, 
                            #num_layer='2')
#'''全连接层'''
#conv_layer2_flat, num_features = flatten_layer(conv_layer2)   # 将最后操作的数据展开
#hidden_layer1 = add_fully_layer(conv_layer2_flat, num_features, 1000, num_layer='1', activation=tf.nn.relu)
#logits = add_fully_layer(hidden_layer1, 1000, num_classes, num_layer='2')


'''RNN网络结构'''
logits = add_RNN_Cell(inputs=x)


predictions = tf.nn.softmax(logits)    
'''loss'''
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
cross_entropy = -tf.reduce_sum(y*tf.log(predictions), reduction_indices=[1])
#square_error = tf.reduce_sum(tf.square(y-logits), reduction_indices=[1])
with tf.name_scope('losses'):
    losses = tf.reduce_mean(cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses)

'''session'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs', sess.graph)  # 将计算图写入文件
