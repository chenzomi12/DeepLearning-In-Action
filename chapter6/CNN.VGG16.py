#!/usr/bin/python
# -*- coding: UTF-8 -*-

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image


input_shape = (3, 224, 224)
img_input = Input(shape=input_shape)

# Block1
x = Convolution2D(64, 3, 3, activation='relu',
                  border_mode='same', name='block1_conv1')(img_input)
x = Convolution2D(64, 3, 3, activation='relu',
                  border_mode='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block2
x = Convolution2D(128, 3, 3, activation='relu',
                  border_mode='same', name='block2_conv1')(x)
x = Convolution2D(128, 3, 3, activation='relu',
                  border_mode='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block3
x = Convolution2D(256, 3, 3, activation='relu',
                  border_mode='same', name='block3_conv1')(x)
x = Convolution2D(256, 3, 3, activation='relu',
                  border_mode='same', name='block3_conv2')(x)
x = Convolution2D(256, 3, 3, activation='relu',
                  border_mode='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block4
x = Convolution2D(512, 3, 3, activation='relu',
                  border_mode='same', name='block4_conv1')(x)
x = Convolution2D(512, 3, 3, activation='relu',
                  border_mode='same', name='block4_conv2')(x)
x = Convolution2D(512, 3, 3, activation='relu',
                  border_mode='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block5
x = Convolution2D(512, 3, 3, activation='relu',
                  border_mode='same', name='block5_conv1')(x)
x = Convolution2D(512, 3, 3, activation='relu',
                  border_mode='same', name='block5_conv2')(x)
x = Convolution2D(512, 3, 3, activation='relu',
                  border_mode='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(1000, activation='softmax', name='predictions')(x)

model = Model(img_input, x)
model.load_weights(weights_path)


def main():


if __name__ == '__main__':
    main()
