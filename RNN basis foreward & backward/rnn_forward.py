import numpy as np
from rnn_cell_forward import rnn_cell_forward 


def rnn_forward(x, a0, parameters): 
    """
    根据图3来实现循环神经网络的前向传播
    
    参数：
        x -- 输入的全部数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为 (n_a, m)
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）
    
    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y_pred -- 所有时间步的预测，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """ 
    # 初始化“caches”，它将以列表类型包含所有的cache 
    caches = [] 
    
    # 获取 x 与 Wya 的维度信息 
    n_x, m, T_x = x.shape 
    n_y, n_a = parameters["Wya"].shape 
    
    # 使用0来初始化“a” 与“y” 
    a = np.zeros([n_a, m, T_x]) 
    y_pred = np.zeros([n_y, m, T_x]) 
    
    # 初始化“next” 
    a_next = a0 
    
    # 遍历所有时间步 
    for t in range(T_x): 
        ## 1.使用rnn_cell_forward函数来更新“next”隐藏状态与cache。 
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters) 
        
        ## 2.使用 a 来保存“next”隐藏状态（第 t ）个位置。 
        a[:, :, t] = a_next 
        
        ## 3.使用 y 来保存预测值。 
        y_pred[:, :, t] = yt_pred 
        
        ## 4.把cache保存到“caches”列表中。 
        caches.append(cache) 
        
    # 保存反向传播所需要的参数 
    caches = (caches, x) 
        
    return a, y_pred, caches


if __name__ == '__main__':
    np.random.seed(1) 
    x = np.random.randn(3,10,4) 
    a0 = np.random.randn(5,10) 
    Waa = np.random.randn(5,5) 
    Wax = np.random.randn(5,3) 
    Wya = np.random.randn(2,5) 
    ba = np.random.randn(5,1) 
    by = np.random.randn(2,1) 
    parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by} 
    a, y_pred, caches = rnn_forward(x, a0, parameters)


    print("a[4][1] = ", a[4][1]) 
    print("a.shape = ", a.shape) 
    print("y_pred[1][3] =", y_pred[1][3]) 
    print("y_pred.shape = ", y_pred.shape) 
    print("caches[1][1][3] =", caches[1][1][3]) 
    print("len(caches) = ", len(caches))
    print(caches)