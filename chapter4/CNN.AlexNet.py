#!/usr/bin/python
# -*- coding: UTF-8 -*-
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Activation, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

input_shape = (224, 224, 3)

# Input Layer
img_input = Input(shape=input_shape)

# Layer1
conv1 = Conv2D(96, (11, 11), strides=(4, 4), activation="relu",
               name="conv1", padding="same")(img_input)
pool1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(conv1)
print(conv1.get_shape)
print(pool1.get_shape)

# Layer2
conv2 = Conv2D(256, (5, 5), activation="relu",
               name="conv2", padding="same")(pool1)
pool2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(conv2)
print(conv2.get_shape)
print(pool2.get_shape)

# Layer3,4
conv3 = Conv2D(384, (3, 3), activation="relu",
               name="conv3", padding="same")(pool2)
conv4 = Conv2D(384, (3, 3), activation="relu",
               name="conv4", padding="same")(conv3)
print(conv3.get_shape)
print(conv4.get_shape)

# Layer5
conv5 = Conv2D(256, (3, 3), activation="relu",
               name="conv5", padding="same")(conv4)
pool5 = MaxPooling2D((3, 3), strides=(3, 3), name='pool5')(conv5)
print(conv5.get_shape)
print(pool5.get_shape)

# Layer6,7
pool5 = Flatten(name='flatten')(pool5)
fc1 = Dense(4096, activation='relu', name='fc1')(pool5)
fc2 = Dense(4096, activation='relu', name='fc2')(fc1)
print(fc1.get_shape)
print(fc2.get_shape)

# Output Layer
otuput = Dense(1, activation='softmax', name='predictions')(fc2)

model = Model(img_input, otuput)

# 优化算法使用 随机梯度下降
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())


batch_size = 16

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'CatDogData/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'CatDogData/validation',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')


# 使用early stopping返回最佳epoch对应的model
early_stopping = EarlyStopping(monitor='val_loss', patience=1)

history_callback = model.fit_generator(
    train_generator,
    epochs=50,
    steps_per_epoch=4002 // batch_size,
    validation_data=validation_generator,
    validation_steps=1200 // batch_size)

pandas.DataFrame(history_callback.history).to_csv("./AlexNet_model.csv")
model.save_weights('./AlexNet_model.h5')


if __name__ == '__main__':
    main()


def get_alexnet(input_shape, nb_classes):
    # code adapted from https://github.com/heuritech/convnets-keras

    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(96, （11, 11）, strides=(4, 4), activation='relu',
                    name='conv_1', init='he_normal')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
        Convolution2D(128, 5, 5, activation="relu", init='he_normal', name='conv_2_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_2)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu',
                           name='conv_3', init='he_normal')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
        Convolution2D(192, 3, 3, activation="relu", init='he_normal', name='conv_4_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_4)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_4")

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
        Convolution2D(128, 3, 3, activation="relu", init='he_normal', name='conv_5_' + str(i + 1))(
            splittensor(ratio_split=2, id_split=i)(conv_5)
        ) for i in range(2)], mode='concat', concat_axis=1, name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1',
                    init='he_normal')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2',
                    init='he_normal')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_classes, name='dense_3_new', init='he_normal')(dense_3)

    prediction = Activation("softmax", name="softmax")(dense_3)

    alexnet = Model(input=inputs, output=prediction)

    return alexnet
