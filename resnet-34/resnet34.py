"""
ResNet-34 model

Original paper:
    https://arxiv.org/pdf/1512.03385.pdf

Implementation by Zach D
"""

import tensorflow as tf


def basic_block(x, filter_size, filters, stride=1, residual=True):
    '''
    Basic residual block with the ability to 
    turn off the residual for downsampling block
    '''
    identity = x
    x = tf.keras.layers.Conv2D(filters, filter_size, strides=stride,
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, filter_size, strides=1,
                               padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if residual:
        x = tf.keras.layers.add([x, identity])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def resnet34(input_shape):
    '''ResNet-34 model'''

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2,
                               padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = basic_block(x, (3, 3), 64)
    x = basic_block(x, (3, 3), 64)
    x = basic_block(x, (3, 3), 64)

    x = basic_block(x, (3, 3), 128, stride=2, residual=False) # Downsample block (no residual)
    x = basic_block(x, (3, 3), 128)
    x = basic_block(x, (3, 3), 128)
    x = basic_block(x, (3, 3), 128)

    x = basic_block(x, (3, 3), 256, stride=2, residual=False) # Downsample block (no residual)
    x = basic_block(x, (3, 3), 256)
    x = basic_block(x, (3, 3), 256)
    x = basic_block(x, (3, 3), 256)
    x = basic_block(x, (3, 3), 256)
    x = basic_block(x, (3, 3), 256)

    x = basic_block(x, (3, 3), 512, stride=2, residual=False) # Downsample block (no residual)
    x = basic_block(x, (3, 3), 512)
    x = basic_block(x, (3, 3), 512)

    x = tf.keras.layers.AveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1000, activation='softmax')(x)

    resnet = tf.keras.Model(inputs, outputs)
    resnet.summary()
    return resnet

if __name__ == '__main__':
    model = resnet34(input_shape=(224, 224, 3))