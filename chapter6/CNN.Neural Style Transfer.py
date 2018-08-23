#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
# 图像风格转换

风格转换就是生成一张图片X内容上与图像P的内容相似, 但是风格上与图像A相似. 通过优化
损失函数实现，其中损失函数由三部分组成: “风格损失”, “内容损失”, “总损失”:

- 总变化损失代表图像中像素与像素之间的局部空间连续性;

- 风格损失是从基本图像P上面提取的Gram矩阵和从不同的网络层中提取的卷积图之间的L2距离的和,
- 其基本思想是从不同空间信息上提取风格图像A的颜色和纹理;

- 内容损失是在基本图像P从高维网络层提取到的特征和合成图像X的特征之间的L2距离,
- 这样可以使得最后的生成图像尽量与基本图像P接近.

# [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''
import time
import numpy as np
from scipy.misc import imsave
from scipy.optimize import fmin_l_bfgs_b

from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16
from keras import backend as K


def preprocess_image(img_path, img_size):
    '''
    预处理输入图片的程序

    打开图像后把图片resize成定义大小并返回keras中vgg16的规格
    open, resize and format pictures into appropriate tensors
    '''
    img = load_img(img_path, target_size=img_size)
    img = img_to_array(img)  # h * w * 3
    img = np.expand_dims(img, axis=0)  # 1 * h * w * 3
    img = vgg16.preprocess_input(img)  # 对img像素值减去训练图片的平均像素值
    return img  # 1 * h * w * 3


def deprocess_image(x, target_size):
    '''
    最后生成输出图片的处理程序

    '''
    x = x.reshape((target_size[0], target_size[1], 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gram_matrix(x):
    '''
    gram矩阵--特征图的外积矩阵
    度量各个维度自己的特性以及各个维度之间的关系
    '''
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style_features, comb_features, target_size):
    '''
    风格损失(style loss)的目的是在生成的图像中保持参考图像的风格
    其中对于风格的定义使用了风格和生成特征图的gram矩阵
    '''
    assert K.ndim(style_features) == 3
    assert K.ndim(comb_features) == 3

    channels = 3
    S = gram_matrix(style_features)
    C = gram_matrix(comb_features)
    size = target_size[0] * target_size[1]  # w * h
    return K.sum(K.square(S - C)) / (4. * pow(channels, 2) * pow(size, 2))


def content_loss(base_features, comb_features):
    ''' 内容损失(content loss)函数 目的是在生成的图像中保持图像的内容
    '''
    return K.sum(K.square(comb_features - base_features))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x, size):
    assert K.ndim(x) == 4
    row = size[0] - 1
    col = size[1] - 1
    a = K.square(x[:, :row, :col, :] - x[:, 1:, :col, :])
    b = K.square(x[:, :row, :col, :] - x[:, :row, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def eval_loss_and_grads(x, size, f_outputs):
    '''
    计算损失和梯度
    '''
    x = x.reshape((1, size[0], size[1], 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):
    '''
    Evaluator 类从两个不同的程序中分别获得loss损失和gradients梯度，然后统一计算
    '''

    def __init__(self, size, f_outputs):
        self.loss_value = None
        self.grads_values = None
        self.size = size
        self.f_outputs = f_outputs

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(
            x, self.size, self.f_outputs)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def main():
    base_img_path = "img/base_img.jpg"
    style_img_path = "img/style_img.jpg"
    target_prefix = "target"
    iterations = 10

    # 不同的损失权重
    total_variation_weight = 1.0  # 总变化损失权重
    style_weight = 1.0  # 风格损失权重
    content_weight = 0.025  # 内容损失权重

    # 确定风格转移图的尺寸 norws: h, ncols: w
    img_nrows = 400
    width, height = load_img(base_img_path).size
    img_ncols = int(width * img_nrows / height)
    target_size = (img_nrows, img_ncols)

    # 实例化输入图片为keras的tensor对象 -> shape=(3, h, w, 3)
    base_img = K.variable(preprocess_image(base_img_path, target_size))
    style_img = K.variable(preprocess_image(style_img_path, target_size))

    # 实例化合成的图片为keras的tensor对象 -> shape=(1, h, w, 3)
    comb_img = K.placeholder((1, img_nrows, img_ncols, 3))

    # 把输入的三张图片合并成keras的tensor对象 -> shape=(3, h, w, 3)
    input_tensor = K.concatenate([base_img, style_img, comb_img], axis=0)

    # 建立VGG网络 把base_img, style_img, comb_img 三张图片作为输入
    # 然后载入预先训练好的VGGNet的权重文件
    model = vgg16.VGG16(input_tensor=input_tensor,
                        weights='imagenet', include_top=False)

    for layer in model.layers:
        print("[layer name]: %14s, [Outpu]: %s" % (layer.name, layer.output))
    print("Model loaded finish")

    # 把VGG网络中的每一层的命名和输出信息放入字典里
    vgg_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # 定义内容损失
    # 把block4 conv2层后的特征作为"内容损失函数"
    loss = K.variable(0.)
    # Relu_8:0 shape=(3, 50, 66, 512)
    layer_features = vgg_dict['block4_conv2']
    base_img_features = layer_features[0, :, :, :]
    comb_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(base_img_features, comb_features)

    feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                      'block4_conv1', 'block5_conv1']

    # 定义风格损失
    # 遍历feature_layers卷积层，把其征作图为"内容损失函数"
    for layer_name in feature_layers:
        layer_features = vgg_dict[layer_name]
        style_features = layer_features[1, :, :, :]
        comb_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, comb_features, target_size)
        loss += (style_weight / len(feature_layers)) * sl  # 0.25

    # 定义总损失
    loss += total_variation_weight * \
        total_variation_loss(comb_img, target_size)

    # 返回loss函数关于comb_img的梯度
    grads = K.gradients(loss, comb_img)

    outputs = [loss]
    outputs += grads
    f_outputs = K.function([comb_img], outputs)  # 实例化一个Keras函数
    evaluator = Evaluator(target_size, f_outputs)

    # 创建一个归一化的(1, h, w, 3)图像
    x = np.random.uniform(0, 255, (1, target_size[0], target_size[1], 3))
    x -= 128.

    # run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss
    for i in range(iterations):
        print'Start iteration', i
        start_time = time.time()

        cur_loss = evaluator.loss
        cur_grads = evaluator.grads
        # 使用scip的L-BFGS算法计算损失函数最小值
        x, min_val, info = fmin_l_bfgs_b(
            cur_loss, x.flatten(), fprime=cur_grads, maxfun=20)

        end_time = time.time()

        img = deprocess_image(x.copy(), target_size)
        fname = target_prefix + '_%d.png' % i
        imsave(fname, img)
        print('Current loss value:', min_val)
        print('Image saved as', fname)
        print'Iteration %d completed in %ds' % (i, end_time - start_time)


if __name__ == '__main__':
    main()
