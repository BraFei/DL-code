### 手势识别 ###
#### 整体框架 ####
Linear->Relu->Linear->Relu->Linear
#### 问答 ####
Q：性能怎么样 <br>
A: 训练集： 99%， 测试集： 72% <br>
![Result](https://i.imgur.com/sT4LG4W.png)

Q: 预处理怎么做的<br>
A：只有正则化，dropout没有做<br>
Q：数据存储和展示是怎么实现的<br>
A: 使用Save类对数据进行存储和取出 <br>
