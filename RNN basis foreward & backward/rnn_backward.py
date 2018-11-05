import numpy as np
from rnn_forward import rnn_forward
from rnn_cell_backward import rnn_cell_backward


def rnn_backward(da, caches):
    """
    在整个输入数据序列上实现RNN的反向传播
    
    参数：
        da -- 所有隐藏状态的梯度，维度为(n_a, m, T_x)
        caches -- 包含向前传播的信息的元组
    
    返回：    
        gradients -- 包含了梯度的字典：
                        dx -- 关于输入数据的梯度，维度为(n_x, m, T_x)
                        da0 -- 关于初始化隐藏状态的梯度，维度为(n_a, m)
                        dWax -- 关于输入权重的梯度，维度为(n_a, n_x)
                        dWaa -- 关于隐藏状态的权值的梯度，维度为(n_a, n_a)
                        dba -- 关于偏置的梯度，维度为(n_a, 1)
    """
    # 从caches中获取第一个cache（t=1）的值
    caches, x = caches
    a1, a0, x1, parameters = caches[0]
    
    # 获取da与x1的维度信息
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # 初始化梯度
    dx = np.zeros([n_x, m, T_x])
    dWax = np.zeros([n_a, n_x])
    dWaa = np.zeros([n_a, n_a])
    dba = np.zeros([n_a, 1])
    da0 = np.zeros([n_a, m])
    da_prevt = np.zeros([n_a, m])
    print(caches)
    print(caches[0][0])
    print(caches[0][1])
    print(caches[0][2])
    print(caches[0][3])
    # 处理所有时间步
    for t in reversed(range(T_x)):
        # 计算时间步“t”时的梯度
        gradients = rnn_cell_backward(da[:, :, t] + da_prevt, caches[t])
        
        #从梯度中获取导数
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        
        # 通过在时间步t添加它们的导数来增加关于全局导数的参数
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    #将 da0设置为a的梯度，该梯度已通过所有时间步骤进行反向传播
    a0 = da_prevt

    #保存这些梯度到字典内
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}

    return gradients


if __name__ == '__main__':
	np.random.seed(1)
	x = np.random.randn(3,10,4)
	a0 = np.random.randn(5,10)
	Wax = np.random.randn(5,3)
	Waa = np.random.randn(5,5)
	Wya = np.random.randn(2,5)
	ba = np.random.randn(5,1)
	by = np.random.randn(2,1)
	parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
	a, y, caches = rnn_forward(x, a0, parameters)
	da = np.random.randn(5, 10, 4)
	gradients = rnn_backward(da, caches)

	print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
	print("gradients[\"dx\"].shape =", gradients["dx"].shape)
	print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
	print("gradients[\"da0\"].shape =", gradients["da0"].shape)
	print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
	print("gradients[\"dWax\"].shape =", gradients["dWax"].shape)
	print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
	print("gradients[\"dWaa\"].shape =", gradients["dWaa"].shape)
	print("gradients[\"dba\"][4] =", gradients["dba"][4])
	print("gradients[\"dba\"].shape =", gradients["dba"].shape)
