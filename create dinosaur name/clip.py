import numpy as np


def clip(gradients, maxValue):
    """
    使用maxValue来修剪梯度
    
    参数：
        gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内
        
    返回：
        gradients -- 修剪后的梯度
    """
    # 获取参数
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    
    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients

if __name__ == '__main__':
    np.random.seed(3)
    dWax = np.random.randn(5,3)*10
    dWaa = np.random.randn(5,5)*10
    dWya = np.random.randn(2,5)*10
    db = np.random.randn(5,1)*10
    dby = np.random.randn(2,1)*10
    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
    gradients = clip(gradients, 10)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"db\"][4] =", gradients["db"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
