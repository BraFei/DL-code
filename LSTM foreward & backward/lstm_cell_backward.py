import numpy as np

from lstm_cell_forward import lstm_cell_forward

def lstm_cell_backward(da_next, dc_next, cache):
    """
    实现LSTM的单步反向传播
    
    参数：
        da_next -- 下一个隐藏状态的梯度，维度为(n_a, m)
        dc_next -- 下一个单元状态的梯度，维度为(n_a, m)
        cache -- 来自前向传播的一些参数
        
    返回：
        gradients -- 包含了梯度信息的字典：
                        dxt -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 先前的隐藏状态的梯度，维度为(n_a, m)
                        dc_prev -- 前的记忆状态的梯度，维度为(n_a, m, T_x)
                        dWf -- 遗忘门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbf -- 遗忘门的偏置的梯度，维度为(n_a, 1)
                        dWi -- 更新门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbi -- 更新门的偏置的梯度，维度为(n_a, 1)
                        dWc -- 第一个“tanh”的权值的梯度，维度为(n_a, n_a + n_x)
                        dbc -- 第一个“tanh”的偏置的梯度，维度为(n_a, n_a + n_x)
                        dWo -- 输出门的权值的梯度，维度为(n_a, n_a + n_x)
                        dbo -- 输出门的偏置的梯度，维度为(n_a, 1)
    """
    # 从cache中获取信息
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    # 获取xt与a_next的维度信息
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    # 根据公式7-10来计算门的导数
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
    dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
    dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)
    
    # 根据公式11-14计算参数的导数
    concat = np.concatenate((a_prev, xt), axis=0).T
    dWf = np.dot(dft, concat)
    dWi = np.dot(dit, concat)
    dWc = np.dot(dcct, concat)
    dWo = np.dot(dot, concat)
    dbf = np.sum(dft,axis=1,keepdims=True)
    dbi = np.sum(dit,axis=1,keepdims=True)
    dbc = np.sum(dcct,axis=1,keepdims=True)
    dbo = np.sum(dot,axis=1,keepdims=True)
    
    
    # 使用公式15-17计算洗起来了隐藏状态、先前记忆状态、输入的导数。
    da_prev = np.dot(parameters["Wf"][:, :n_a].T, dft) + np.dot(parameters["Wc"][:, :n_a].T, dcct) +  np.dot(parameters["Wi"][:, :n_a].T, dit) + np.dot(parameters["Wo"][:, :n_a].T, dot)
        
    dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
    
    dxt = np.dot(parameters["Wf"][:, n_a:].T, dft) + np.dot(parameters["Wc"][:, n_a:].T, dcct) +  np.dot(parameters["Wi"][:, n_a:].T, dit) + np.dot(parameters["Wo"][:, n_a:].T, dot)
    
    # 保存梯度信息到字典
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
                "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
    
    return gradients


if __name__ == '__main__':
    np.random.seed(1)
    xt = np.random.randn(3,10)
    a_prev = np.random.randn(5,10)
    c_prev = np.random.randn(5,10)
    Wf = np.random.randn(5, 5+3)
    bf = np.random.randn(5,1)
    Wi = np.random.randn(5, 5+3)
    bi = np.random.randn(5,1)
    Wo = np.random.randn(5, 5+3)
    bo = np.random.randn(5,1)
    Wc = np.random.randn(5, 5+3)
    bc = np.random.randn(5,1)
    Wy = np.random.randn(2,5)
    by = np.random.randn(2,1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    da_next = np.random.randn(5,10)
    dc_next = np.random.randn(5,10)
    gradients = lstm_cell_backward(da_next, dc_next, cache)
    print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
    print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
    print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
    print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
    print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
    print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
