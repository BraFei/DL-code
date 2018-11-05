import numpy as np
import cllm_utils
from clip import clip

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01): 
	""" 
	执行训练模型的单步优化。 
	参数： X -- 整数列表，其中每个整数映射到词汇表中的字符。 
	Y -- 整数列表，与X完全相同，但向左移动了一个索引。 
	a_prev -- 上一个隐藏状态 
	parameters -- 字典，包含了以下参数： 
		Wax -- 权重矩阵乘以输入，维度为(n_a, n_x) 
		Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a) 
		Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a) 
		b -- 偏置，维度为(n_a, 1) 
		by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1) 
		learning_rate -- 模型学习的速率 
	返回： 
		loss -- 损失函数的值（交叉熵损失） 
		gradients -- 字典，
		包含了以下参数： 
			dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x) 
			dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a) 
			dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a) 
			db -- 偏置的梯度，维度为(n_a, 1) 
			dby -- 输出偏置向量的梯度，维度为(n_y, 1) 
			a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1) 
	""" 
	# 前向传播 
	loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters) 

	# 反向传播 
	gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache) 

	# 梯度修剪，[-5 , 5] 
	gradients = clip(gradients,5) 

	# 更新参数 
	parameters = cllm_utils.update_parameters(parameters,gradients,learning_rate) 

	return loss, gradients, a[len(X)-1]

if __name__ == '__main__':
	np.random.seed(1)
	vocab_size, n_a = 27, 100
	a_prev = np.random.randn(n_a, 1)
	Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
	b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
	parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
	X = [12,3,5,11,22,3]
	Y = [4,14,11,22,25, 26]

	loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
	print("Loss =", loss)
	print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
	print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
	print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
	print("gradients[\"db\"][4] =", gradients["db"][4])
	print("gradients[\"dby\"][1] =", gradients["dby"][1])
	print("a_last[4] =", a_last[4])
