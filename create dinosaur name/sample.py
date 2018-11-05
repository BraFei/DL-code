import numpy as np
import cllm_utils

def sample(parameters, char_to_is, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样
    
    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典
        seed -- 随机种子
        
    返回：
        indices -- 包含采样字符索引的长度为n的列表。
    """
    
    # 从parameters 中获取参数
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    # 步骤1 
    ## 创建独热向量x
    x = np.zeros((vocab_size,1))
    
    ## 使用0初始化a_prev
    a_prev = np.zeros((n_a,1))
    
    # 创建索引的空列表，这是包含要生成的字符的索引的列表。
    indices = []
    
    # IDX是检测换行符的标志，我们将其初始化为-1。
    idx = -1
    
    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    #（我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
    counter = 0
    newline_character = char_to_ix["\n"]
    
    while (idx != newline_character and counter < 50):
        # 步骤2：使用公式1、2、3进行前向传播
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = cllm_utils.softmax(z)
        
        # 设定随机种子
        np.random.seed(counter + seed)
        
        # 步骤3：从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())
        
        # 添加到索引中
        indices.append(idx)
        
        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        
        # 更新a_prev为a
        a_prev = a 
        
        # 累加器
        seed += 1
        counter +=1
    
    if(counter == 50):
        indices.append(char_to_ix["\n"])
    
    return indices
        
if __name__ == '__main__':
    np.random.seed(2)
    _, n_a = 20, 100
    # 获取名称
    data = open("dinos.txt", "r").read()

    # 转化为小写字符
    data = data.lower()

    # 转化为无序且不重复的元素列表
    chars = list(set(data))

    # 获取大小信息
    data_size, vocab_size = len(data), len(chars)

    char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}

    print(char_to_ix)
    print(ix_to_char)


    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


    indices = sample(parameters, char_to_ix, 0)
    print("Sampling:")
    print("list of sampled indices:", indices)
    print("list of sampled characters:", [ix_to_char[i] for i in indices])
