import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import time
import matplotlib.pyplot as plt  # plt 用于显示图片 
import matplotlib.image as mpimg # mpimg 用于读取图片 
import numpy as np #这是自己拍的图片

import matplotlib.pyplot as plt 
# plt 用于显示图片 
import matplotlib.image as mpimg 
# mpimg 用于读取图片 
import numpy as np

isTrain = False
train_steps = 100
checkpoint_steps = 1
checkpoint_dir = 'Figure_Model/'

my_image1 = "4.png" #定义图片名称 
fileName1 = "images/fingers/" + my_image1 
#图片地址 
image1 = mpimg.imread(fileName1) 
#读取图片 
# plt.imshow(image1) 
#显示图片 
my_image1 = image1.reshape(1,64, 64, 3) 


# 重新构建权重矩阵的维度方便预测使用
W1 = tf.get_variable("W1",[4, 4, 3, 8])
W2 = tf.get_variable("W2", [2, 2, 8, 16])

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        W1 = sess.run(W1)
        W2 = sess.run(W2)

parameters = {
    "W1": W1,
    "W2": W2}
#print(parameters['W1'])
##########################
W1 = tf.convert_to_tensor(parameters["W1"])
W2 = tf.convert_to_tensor(parameters["W2"])
X = tf.placeholder("float", [1, 64, 64, 3])
#Conv2d : 步伐：1，填充方式：“SAME”
Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
#ReLU
A1 = tf.nn.relu(Z1)
#Max pool : 窗口大小：8x8，步伐：8x8，填充方式：“SAME”
P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
#Conv2d : 步伐：1，填充方式：“SAME”
Z2 = tf.nn.conv2d(P1, W2,strides=[1, 1, 1, 1], padding='SAME')
#ReLU ：
A2 = tf.nn.relu(Z2)
#Max pool : 过滤器大小：4x4，步伐：4x4，填充方式：“SAME”
P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")
#一维化上一层的输出
P = tf.contrib.layers.flatten(P2)

#全连接层（FC）：使用没有非线性激活函数的全连接层
Z3 = tf.contrib.layers.fully_connected(P, 6, activation_fn=None)

p = tf.argmax(Z3, 1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
z3 = sess.run(Z3, feed_dict = {X: my_image1})
print(z3)

pp = sess.run(p, feed_dict = {X: my_image1})
print(pp)

prediction = sess.run(p, feed_dict = {X: my_image1})
#开始预测 
print("预测结果: y = " + str(np.squeeze(prediction)))