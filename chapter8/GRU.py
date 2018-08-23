
# coding: utf-8

# In[3]:


import keras
import numpy as np


# In[7]:


word_dim = 10
hidden_dim = 1
c = np.zeros(word_dim)
E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))

def GRU_forward(x_t, h_t_prev):
    # 在simple RNN中前馈计算使用的是
    # h_t = np.tanh(U.dot(x_t) + W.dot(h_t_prev))
    
    # 获取
    x_t_ = E[:,x_t]
       
    # GRU
    z_t = sigmoid(U[0].dot(x_t_) + W[0].dot(h_t_prev) + b[0]) # update gate
    r_t = sigmoid(U[1].dot(x_t_) + W[1].dot(h_t_prev) + b[1]) # reset gate
    c_t = np.tanh(U[2].dot(x_t_) + W[2].dot(h_t_prev * r_t) + b[2])
    h_t = (1 - z_t) * c_t + z_t * h_t_prev
       
    # 最后输出概率
    x_t = softmax(V.dot(h_t) + c)[0]
 
    return [x_t, s_t1]


# In[ ]:


def GRU_gradient(y_pred, y):
    # 向后传播
    loss = categorical_crossentropy(y_pred, y)

    # 计算权重参数梯度
    dE = keras.grad(loss, E)
    dU = keras.grad(loss, U)
    dW = keras.grad(loss, W)
    db = keras.grad(loss, b)
    dV = keras.grad(loss, V)
    dc = keras.grad(loss, c)

