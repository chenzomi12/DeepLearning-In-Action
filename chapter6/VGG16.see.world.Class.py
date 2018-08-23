
# coding: utf-8

# In[1]:

import numpy as np
 
import scipy.misc
import time
import os
import h5py
from scipy.ndimage.filters import gaussian_filter, median_filter

from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:

# path to the model weights file.
weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
 
def VGG_16(w_path=None):
    
    img_input = Input(shape=(3, 224,224))
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dropout(0.5)(x)
    x = Dense(1000, activation='linear', name='predictions')(x) # avoid softmax (see Simonyan 2013)

    model = Model(img_input, x, name='vgg16')
    
    if w_path:
        model.load_weights(w_path)
 
    return model
 
# Creates the VGG models and loads weights
vgg16 = VGG_16(weights_path)
print('Model loaded.')
vgg16.summary()


# In[21]:

def transfer_FCN_Vgg16():
    input_shape = (3, 224, 224)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Convolutional layers transfered from fully-connected layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fc1')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2')(x)
    x = Conv2D(1000, (1, 1), activation='linear', name='predictions_1000')(x)
    #x = Reshape((7,7))(x)

    # Create model
    model = Model(img_input, x)
    weights_path = "fcn_vgg16_weights_tf_dim_ordering_tf_kernels.h5"

    #transfer if weights have not been created
    if os.path.isfile(weights_path) == False:
        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer

        for layer in vgg16.layers:
            weights = layer.get_weights()
            if layer.name=='fc1':
                # weights[0] = np.reshape(weights[0], (7,7,512,4096))
                weights[0] = np.reshape(weights[0], (7,7,512,4096))
            elif layer.name=='fc2':
                weights[0] = np.reshape(weights[0], (1,1,4096,4096))
            elif layer.name=='predictions':
                layer.name='predictions_1000'
                weights[0] = np.reshape(weights[0], (1,1,4096,1000))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        print( 'Successfully transformed!')
    #else load weights
    else:
        model.load_weights(weights_path, by_name=True)
        print( 'Already transformed!')
        
    return model

fcn_vgg16 = transfer_FCN_Vgg16()
print('Model loaded.')
fcn_vgg16.summary()


# In[ ]:

img_width = 224
img_height = 224

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'predictions_1000'

# util function to convert a tensor into a valid image


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# this is the placeholder for the input images
input_img = fcn_vgg16.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in fcn_vgg16.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


"""
imagenet1000_clsid_to_human.txt from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
{
    1: 'goldfish, Carassius aurtus',
    8: 'hen',
    18: 'magpie',
    388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    351: 'hartebeest',
    327: 'starfish, sea star',
    413: 'assault rifle, assault gun',
    672: 'mountain tent',
}
"""
kept_filters = []
all_filter = [1, 8, 18, 388, 351, 327, 413, 672]

for ix, filter_index in enumerate([1]):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 6.

    # we start from a gray image with some random noise
    input_img_data = np.random.random((1, 3, img_width, img_height))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    mFilterSize = 3
    mFilterEvery = 30
    mIter = 1001
    # step 5, mFilterEvery 8, mFilterSize 3

    # we run gradient ascent for 20 steps
    for i in range(mIter):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        input_img_data = np.clip(input_img_data, 0., 255.)
                        
        if mFilterSize is not 0 and i % mFilterEvery == 0 :
            input_img_data = median_filter(input_img_data, size=(1, 1, mFilterSize, mFilterSize))
    
        if i % 50 == 0:
            print('\t%d, Current loss value:%f' % (i, loss_value))

    # decode the resulting input image
    img = deprocess_image(input_img_data[0])
    kept_filters.append((img, loss_value, filter_index))
    end_time = time.time()
    print('%d, Filter %d processed in %ds' % (ix, filter_index, end_time - start_time))
    
print("="*80)
print("Finish!")


# In[68]:

plt.imshow(kept_filters[0][0])
print(kept_filters[0][0].shape)
print(len(kept_filters))


# In[56]:

for img_idx in kept_filters:
    img = img_idx[0]
    scipy.misc.imsave('%s_lr5_it1000_mf510.jpg' % (img_idx[2]), img)


# In[ ]:



