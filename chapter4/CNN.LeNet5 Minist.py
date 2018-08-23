
# coding: utf-8

# In[153]:


import os
import numpy as np
import keras 
import pprint
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from keras.models import Sequential, Model
import keras.backend as K

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # 加载手写字体
# 
# 从keras自带的数据库中加载mnist手写字体

# In[154]:


img_rows, img_cols = (28,28)
num_classes = 10

def get_mnist_data():
    """
    加载mnist手写字体数据集
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = get_mnist_data()
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# # LetNet5 网络模型架构

# In[155]:


def LeNet5(w_path=None):
    
    input_shape = (1, img_rows, img_cols)
    img_input = Input(shape=input_shape)
   
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = Conv2D(64, (3, 3), activation="relu", padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    x = Dropout(0.25)(x)
    
    x = Flatten(name='flatten')(x)
    
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    
    model = Model(img_input, x, name='LeNet5')
    if(w_path): model.load_weights(w_path)
    
    return model

lenet5 = LeNet5()
print('Model loaded.')
lenet5.summary()


# # LeNet5 模型训练

# In[12]:


from keras.callbacks import ModelCheckpoint

if not os.path.exists('lenet5_checkpoints'):
    os.mkdir('lenet5_checkpoints')
    
lenet5.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(monitor='val_acc', 
                             filepath = 'lenet5_checkpoints/model_{epoch:02d}_{val_acc:.3f}.h5',
                             save_best_only = True)

lenet5.fit(x_train, y_train,
         batch_size = 128,
         epochs = 30,
         verbose = 1,
         validation_data = (x_test, y_test),
         callbacks = [checkpoint])

score = lenet5.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[39]:


from sklearn.metrics import classification_report

y_pred = lenet5.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(classification_report(y_test, y_pred))


# # 可视化过程

# In[156]:


lenet5 = LeNet5()
lenet5.compile(optimizer='adam', 
               loss='categorical_crossentropy', 
               metrics=['accuracy'])

lenet5.summary()

x_train, y_train, x_test, y_test = get_mnist_data()
test_loss, test_acc = lenet5.evaluate(x_test, y_test, verbose=1, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[157]:


i = np.random.choice(x_test.shape[0])
print(i)
plt.imshow(x_test[i, 0], interpolation='None', cmap='gray')
print("{} label:{}".format(i, y_train[i,:]))


# In[158]:


layer1 = lenet5.get_layer("conv1")


w = layer1.get_weights()
print(w[0].shape)
print(np.squeeze(w[0]).shape)

# plt.figure(figsize=(15,15))
# plt.title("Conv1 weights")


# In[181]:


def get_activations(model, model_input, layer_name = None):
    activations = []
    inp = [model.input]
    
    # 所有层的输出 
    model_layers = [layer.output for layer in model.layers if 
               layer.name == layer_name or layer_name is None]
    pprint.pprint(model_layers)
    
    newmodel_layers = []
    newmodel_layers.append(model_layers[0])
    newmodel_layers.append(model_layers[1])
    newmodel_layers.append(model_layers[2])
    newmodel_layers.append(model_layers[3])
    newmodel_layers.append(model_layers[4])
    newmodel_layers.append(model_layers[7])
    newmodel_layers.append(model_layers[9])
    newmodel_layers.append(model_layers[11])

    funcs = [K.function(inp + [K.learning_phase()], [layer]) for layer in newmodel_layers]
    
    list_inputs = model_input.reshape(1,1,28,28)
    print(list_inputs.shape)
    
        
    layer_outputs = [func([list_inputs]) for func in funcs]
    for activation in layer_outputs:
        activations.append(activation)
        
    return activations

activations = get_activations(lenet5, x_test[i])


# In[306]:


def display_tensor(tensors):
    if tensors.shape == (1, 1, 28, 28):
        tt = np.hstack(np.transpose(tensors[0], (0, 1, 2)))
        plt.figure(figsize=(15,1))
        plt.imshow(tt, interpolation='None', cmap='gray')
        plt.show()
    else:
        tt = np.hstack(np.transpose(tensors[0], (0, 1, 2)))
        plt.figure(figsize=(15,1))
        plt.imshow(tt, interpolation='None', cmap='hot')
        plt.show()        
        
def display_matrix(matrix):
    num_activations = len(matrix)
    tt = np.repeat(matrix, 10, axis=0)
    
    plt.figure(figsize=(20,2))
    plt.imshow(tt, interpolation='None', cmap='hot')
    plt.colorbar()
    plt.show()

    
def display_activations(activations_tensor):
    img_size = activations_tensor[0][0].shape[0]
    assert img_size == 1, 'One image at a time to visualize!'

    for i, activation_map in enumerate(activations_tensor):
        print('Displaying activation: {} {}'.format(i, activation_map[0].shape))
        activation_map = activation_map[0]
        shape = activation_map.shape
        if len(shape) == 4:
            display_tensor(activation_map)
            pass
        if len(shape) == 2:
            display_matrix(activation_map)
        
display_activations(activations)

