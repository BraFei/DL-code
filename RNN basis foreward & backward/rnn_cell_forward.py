import numpy as np
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rnn_cell_forward(xt, a_prev, parameters):
    """
    参数：
        xt -- 时间步“t”输入的数据，维度为（n_x, m）
        a_prev -- 时间步“t - 1”的隐藏隐藏状态，维度为（n_a, m）
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a_next -- 下一个隐藏状态，维度为（n_a， m）
        yt_pred -- 在时间步“t”的预测，维度为（n_y， m）
        cache -- 反向传播需要的元组，包含了(a_next, a_prev, xt, parameters)
    """
    # 从“parameters”获取参数 
    Wax = parameters["Wax"] 
    Waa = parameters["Waa"] 
    Wya = parameters["Wya"] 
    ba = parameters["ba"] 
    by = parameters["by"]
    
    # 使用上面的公式计算下一个激活值 
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba) 
    
    # 使用上面的公式计算当前单元的输出 
    yt_pred = softmax(np.dot(Wya, a_next) + by) 
    
    # 保存反向传播需要的值 
    cache = (a_next, a_prev, xt, parameters) 
    
    return a_next, yt_pred, cache 


if __name__ == '__main__':
	np.random.seed(1)
	xt = np.random.randn(3, 10)
	a_prev = np.random.randn(5, 10)
	Waa = np.random.randn(5, 5)
	Wax = np.random.randn(5, 3)
	Wya = np.random.randn(2, 5)
	ba = np.random.randn(5, 1)
	by = np.random.randn(2, 1)
	parameters = {'Waa': Waa, 'Wax': Wax, 'Wya': Wya, 'ba': ba, 'by': by}

	a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
	print('a_next[4] = ', a_next[4])
	print('a_next.shape = ', a_next.shape)
	print('yt_pred[1] = ', yt_pred[1])
	print('yt_pred.shape = ', yt_pred.shape)