#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn import linear_model
from sklearn import datasets
import sklearn
import numpy as np
import matplotlib.pyplot as plt

# plt.scatter(data[:, 0], data[:, 1], s=50, c=labels,
#             cmap=plt.cm.Spectral, edgecolors="#313131")


class Config:
    input_dim = 2  # 输入的维度
    output_dim = 2  # 输出的分类数

    epsilon = 0.01  # 梯度下降学习速度
    reg_lambda = 0.01  # 正则化强度


def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)  # 300个数据点，噪声设定0.3
    return X, y


def display_model(model):
    print("W1 {}: \n{}\n".format(model['W1'].shape, model['W1']))
    print("b1 {}: \n{}\n".format(model['b1'].shape, model['b1']))
    print("W2 {}: \n{}\n".format(model['W2'].shape, model['W2']))
    print("b1 {}: \n{}\n".format(model['b2'].shape, model['b2']))


def plot_decision_boundary(pred_func, data, labels):
    '''绘制分类边界图'''
    # 设置最大值和最小值并增加0.5的边界（0.5 padding）
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    h = 0.01

    # 生成一个点阵网格，点阵间距离为h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测整个网格当中的函数值
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # 绘制轮廓和训练样本
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(data[:, 0], data[:, 1], s=40, c=labels, cmap=plt.cm.Spectral)
    plt.show()


def calculate_loss(model, X, y):
    '''
    损失函数
    '''
    num_examples = len(X)  # 训练集大小
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播计算预测值
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算损失值
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # 对损失值进行归一化（可以不加）
    data_loss += Config.reg_lambda / 2 * \
        (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss


def predict(model, x):
    '''
    预测函数
    '''
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 向前传播
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def ANN_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    '''
    网络学习函数，并返回网络
    - nn_hdim: 隐层的神经元节点（隐层的数目）
    - num_passes: 梯度下降迭代次数
    - print_loss: 是否显示损失函数值
    '''
    num_examples = len(X)  # 训练的数据集
    model = {}  # 模型存储定义

    # 随机初始化参数
    np.random.seed(0)
    W1 = np.random.randn(Config.input_dim, nn_hdim) / np.sqrt(Config.input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, Config.output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, Config.output_dim))
    # display_model({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

    # 批量梯度下降
    for i in xrange(0, num_passes + 1):
        # 向前传播
        z1 = X.dot(W1) + b1  # M_200*2 .* M_2*3 --> M_200*3
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2  # M_200*3 .* M_3*2 --> M_200*2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 向后传播
        delta3 = probs  # 得到的预测值
        delta3[range(num_examples), y] -= 1  # 预测值减去实际值
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW2 = (a1.T).dot(delta3)  # W2的导数
        db2 = np.sum(delta3, axis=0, keepdims=True)  # b2的导数
        dW1 = np.dot(X.T, delta2)  # W1的导数
        db1 = np.sum(delta2, axis=0)  # b1的导数

        # 添加正则化项
        dW1 += Config.reg_lambda * W1
        dW2 += Config.reg_lambda * W2

        # 根据梯度下降值更新权重
        W1 += -Config.epsilon * dW1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dW2
        b2 += -Config.epsilon * db2

        # 把新的参数加入模型当中
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print("Loss after iteration %i: %f" %
                  (i, calculate_loss(model, X, y)))

    return model


def main():
    X, y = generate_data()
    model = ANN_model(X, y, 3, print_loss=True)  # 建立三个神经元的隐层
    print display_model(model)

    plot_decision_boundary(lambda x: predict(model, x), X, y)
    plt.title("Logistic Regression")

if __name__ == '__main__':
    main()
