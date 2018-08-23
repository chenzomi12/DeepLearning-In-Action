# 定义神经网络的模型架构 [input, hidden, output]
network_sizes = [3, 4, 2]

# 初始化该神经网络的参数
sizes = network_sizes
num_layers = len(sizes)
biases = [np.random.randn(h, 1) for h in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


def loss_der(network_y, real_y):
    """
    返回损失函数的偏导，损失函数使用 MSE
    L = 1/2(network_y-real_y)^2
    delta_L = network_y-real_y
    """
    return (network_y - real_y)


def sigmoid(z):
    """激活函数使用 sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_der(z):
    """sigmoid函数的导数 derivative of sigmoid."""
    return sigmoid(z) * (1 - sigmoid(z))


def backprop(x, y):
    """
    根据损失函数 C通过反向传播算法返回
    """
    """Return a tuple "(nabla_b, nabla_w)" representing the
    gradient for the cost function C_x.  "nabla_b" and
    "nabla_w" are layer-by-layer lists of numpy arrays, similar
    to "self.biases" and "self.weights"."""

    # 初始化网络参数的导数 权重w的偏导和偏置b的偏导
    delta_w = [np.zeros(w.shape) for w in weights]
    delta_b = [np.zeros(b.shape) for b in biases]

    # 向前传播 feed forward
    activation = x     # 把输入的数据作为第一次激活值
    activations = [x]  # 存储网络的激活值
    zs = []            # 存储网络的加权输入值 (z=wx+b)

    for w, b in zip(weights, biases):
        z = np.dot(w, activation) + b
        activation = sigmoid(z)

        activations.append(activation)
        zs.append(z)

    # 反向传播 back propagation
    # BP1 计算输出层误差
    delta_L = loss_der(activations[-1], y) * sigmoid_der(zs[-1])
    # BP3 损失函数在输出层关于偏置的偏导
    delta_b[-1] = delta_L
    # BP4 损失函数在输出层关于权值的偏导
    delta_w[-1] = np.dot(delta_L, activations[-2].transpose())

    delta_l = delta_L
    for l in range(2, num_layers):
        # BP2 计算第l层误差
        z = zs[-l]
        sp = sigmoid_der(z)
        delta_l = np.dot(weights[-l + 1].transpose(), delta_l) * sp
        # BP3 损失函数在l层关于偏置的偏导
        delta_b[-l] = delta_l
        # BP4 损失函数在l层关于权值的偏导
        delta_w[-l] = np.dot(delta_l, activations[-l - 1].transpose())

    return (delta_w, delta_b)


training_x = np.random.rand(3).reshape(3, 1)
training_y = np.array([0, 1]).reshape(2, 1)
print("training data x:{}, training data y:{}\n".format(training_x, training_y))
backprop(training_x, training_y)
