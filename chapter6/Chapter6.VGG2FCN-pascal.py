
# coding: utf-8

# In[27]:


import numpy as np
 
import scipy.misc
import time
import os
import h5py
from scipy.ndimage.filters import gaussian_filter, median_filter

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Deconv2D,MaxPooling2D,Conv2DTranspose
from keras.layers import Flatten, Dense, Dropout, UpSampling2D, merge, Cropping2D
from keras import backend as K


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Upsample Layer

# In[28]:


from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class Upsimple(Layer):
    """
    Upsample层
    """
    def __init__(self, size=(1,1), target_size=None, **kwargs):
        self.target_size = tuple(target_size) if target_size is not None else None
        self.size = tuple(size) if size is not None else None
        
        self.data_format = K.image_data_format()
        assert self.data_format in {'channels_last', 'channels_first'},         'data_format not in {tf, th}'
    
        super(Upsimple, self).__init__(**kwargs)
    
    def call(self, x):
        if self.target_size is not None:
            x = self.__resize_bilinear_with_target(x,
                                        target_size = self.target_size,
                                        data_format = self.data_format)
        else:
            x = self.__resize_bilinear_with_factor(x,
                                        factor_size = self.size,
                                        data_format = self.data_format)
        return x
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0], input_shape[1], width, height)
        else:
            raise Exception('Invalid data_format: ' + self.data_format)
                
    def __resize_bilinear_with_target(self, x, target_size=None, data_format=None):
        """
        upsimple input image with bilinear method by setup target.
        """
        target_height = target_size[0]
        target_width = target_size[1]
        if data_format == 'channels_first':
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
            X = K.permute_dimensions(x, [0, 2, 3, 1])
            X = tf.image.resize_bilinear(X, new_shape)
            X = K.permute_dimensions(X, [0, 3, 1, 2])
            return X
        else:
            raise Exception("Invilid data format", data_format)
    
    def __resize_bilinear_with_factor(self, x, factor_size=None, data_format=None):
        """
        upsimple input iamge with bilinear method by number factor.
        """
        height_factor = factor_size[0]
        width_factor = factor_size[1]
        if data_format == 'channels_first':
            height = x.shape[2] * height_factor
            width = x.shape[3] * width_factor
            new_shape = tf.constant(np.array([height, width]).astype('int32'))
            X = K.permute_dimensions(x, [0, 2, 3, 1])
            X = tf.image.resize_bilinear(X, new_shape)
            X = K.permute_dimensions(X, [0, 3, 1, 2])
            return X
        else:
            raise Exception("invilid data format", data_format)

# X = Upsimple(size=(32,32))(fcn_vgg16)


# In[29]:


def FCN_Vgg32(input_shape=None):

    if input_shape:
        img_input = Input(shape=input_shape)
    else:
        input_shape = (3, 224, 224)
        img_input = Input(shape=input_shape)

    target_height, target_width = input_shape[1], input_shape[2]

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='conv1_1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv4_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='conv5_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc6')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(21, (1, 1), activation='linear',
               padding='same', name='score_fr')(x)

    conv_size = x.shape[2].value
    deconv_size = (conv_size - 1) * 2 + 4
    extra_size = (deconv_size - conv_size * 2) / 2

    x = Conv2DTranspose(21, (4, 4), activation=None,
                        name="score2", strides=(2, 2), padding="same")(x)

    model = Model(img_input, x)

    return model


fcn_vgg32 = FCN_Vgg32(input_shape=(3, 512, 512))
print('Model loaded.')
fcn_vgg32.summary()


# In[30]:


def fcn_32to16(fcn32model=None):
    if fcn32model is None:
        raise Exception("the mode should not be null.")

    fcn32size = fcn32model.layers[-1].output_shape[2]
    if fcn32size != 32:
        raise Exception("other size is not been format.")

    sp4_ = Conv2D(21, (1, 1), padding="same", activation=None, name='score_pool4')
    sp4 = sp4_(fcn32model.layers[14].output)
    sp5 = fcn32model.layers[-1].output
    
    sum_sp = merge([sp4, sp5], mode='sum')
    ups = Conv2DTranspose(21, (32, 32), activation=None,
                            name="upsample_new", strides=(16, 16), 
                            padding="valid")(sum_sp)

    crop_margin = Cropping2D(cropping=((8, 8),(8, 8)))
    
    model = Model(fcn32model.input, crop_margin(ups))
    return model


fcn_vgg16 = fcn_32to16(fcn_vgg32)
fcn_vgg16.summary()


# In[6]:


test_image = np.ones((3,512,512))
test_image = np.expand_dims(test_image, axis=0)

fcn_vgg16.predict(test_image).shape


# In[7]:


from scipy.io import loadmat

data = loadmat('pascal-fcn16s-dag.mat', matlab_compatible=False, struct_as_record=False)
layers = data['layers']
params = data['params']
description = data['meta'][0][0].classes[0, 0].description


# In[10]:


class2index = {}
for i, classname in enumerate(description[0,:]):
    class2index[str(classname[0])] = i

class2index


# In[21]:


for i in range(0, params.shape[1]-1, 2):
    print(i,
          str(params[0,i].name[0]), params[0,i].value.shape,
          str(params[0,i+1].name[0]),params[0,i+1].value.shape)


# In[15]:


for i in range(layers.shape[1]):
    print(i,
          str(layers[0,i].name[0]), str(layers[0,i].type[0]),
          [str(n[0]) for n in layers[0,i].inputs[0,:]],
          [str(n[0]) for n in layers[0,i].outputs[0,:]])


# In[37]:


layer_names = [layer.name for layer in fcn_vgg16.layers]
print(layer_names)

for i in range(0, params.shape[1]-1, 2):
    matname = '_'.join(params[0, i].name[0].split('_')[0:-1])
    print(matname)
    if matname in layer_names:
        key_idx = layer_names.index(matname)
        print("found:", (matname, key_idx))
        
        layer_w = params[0,i].value
        layer_b = params[0,i+1].value
        print(layer_w.shape)
#         flayer_w = layer_w.transpose((3,2,0,1))
        flayer_w = layer_w
        flayer_w = np.flip(flayer_w, 2)
        flayer_w = np.flip(flayer_w, 3)
        
        print(flayer_w.shape)
        print(fcn_vgg16.layers[key_idx].get_weights()[0].shape)
        assert (flayer_w.shape == fcn_vgg16.layers[key_idx].get_weights()[0].shape)
        assert (layer_b.shape[1] == 1)
        assert (layer_b[:,0].shape == fcn_vgg16.layers[key_idx].get_weights()[1].shape)
        assert (len(fcn_vgg16.layers[key_idx].get_weights()) == 2)
        
        fcn_vgg16.layers[key_idx].set_weights([layer_w, layer_b[:,0]])


# In[44]:


from PIL import Image
from keras.preprocessing import image 
from keras.applications.imagenet_utils import preprocess_input

def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(512,512))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
#     img_tensor /= 255.
    
    print()
    
    if show:
        show_img = np.transpose(img_tensor[0], (1,2,0))
        plt.imshow(show_img)
        plt.show()

    return img_tensor
        
image = load_image("image/2007_000129.jpg")
print(image.shape)
pred = fcn_vgg16.predict(image, batch_size=1)
print("predict finished.")


# In[45]:


a = np.squeeze(pred)
print(a.shape)
b = np.argmax(a, axis=0).astype(np.uint8)
print(b.shape)
print(b)


# In[46]:


from keras.preprocessing import image 

image = image.load_img("image/2007_000129.jpg", target_size=(512,512))
plt.imshow(image)
plt.show()
# b.palette = label.palette
plt.imshow(b)
plt.show()


# In[43]:


fcn_vgg16.save_weights('vgg16_weights_tf_dim_ordering_tf_kernels_pascal_fcn16s.h5')

