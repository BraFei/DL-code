import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time
import matplotlib.pyplot as plt  # plt 用于显示图片 
import matplotlib.image as mpimg # mpimg 用于读取图片 
import numpy as np #这是自己拍的图片

import matplotlib.pyplot as plt 
# plt 用于显示图片 
import matplotlib.image as mpimg 
# mpimg 用于读取图片 
import numpy as np
import tf_utils

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
    Z1 = tf.add(tf.matmul(W1,X),b1)        # Z1 = np.dot(W1, X) + b1
    #Z1 = tf.matmul(W1,X) + b1             #也可以这样写
    A1 = tf.nn.relu(Z1)                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3


isTrain = False
train_steps = 100
checkpoint_steps = 1
checkpoint_dir = 'Figure_Model/'

my_image1 = "5.png" #定义图片名称 
fileName1 = "images/fingers/" + my_image1 
#图片地址 
image1 = mpimg.imread(fileName1) 
#读取图片 
plt.imshow(image1) 
#显示图片 
my_image1 = image1.reshape(1,64 * 64 * 3).T 


W1 = tf.get_variable("W1",[25,12288])
b1 = tf.get_variable("b1",[25,1])
W2 = tf.get_variable("W2", [12, 25])
b2 = tf.get_variable("b2", [12, 1])
W3 = tf.get_variable("W3", [6, 12])
b3 = tf.get_variable("b3", [6, 1])

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        W1 = sess.run(W1)
        b1 = sess.run(b1)
        W2 = sess.run(W2)
        b2 = sess.run(b2)
        W3 = sess.run(W3)
        b3 = sess.run(b3)
params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
 
    
    
    
#这是博主自己拍的图片 
my_image1 = "5.png" 
#定义图片名称 
fileName1 = "images/fingers/" + my_image1 
#图片地址 
image1 = mpimg.imread(fileName1) 
#读取图片 
plt.imshow(image1) 
#显示图片 
my_image1 = image1.reshape(1,64 * 64 * 3).T 
#重构图片



my_image_prediction =predict(my_image1, params) 
#开始预测 
print("预测结果: y = " + str(np.squeeze(my_image_prediction)))
    
    
    
    
    
    
    
    
    
    