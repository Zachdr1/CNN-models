"""
LeNet-5 model for MNIST dataset.

GoogLeNet architecture adapted for CIFAR-10 dataset classification. Original paper:
    https://arxiv.org/pdf/1409.4842.pdf

Zach D.
"""
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

_EPOCHS = 14
_NUM_CLASSES = 10
_BATCH_SIZE = 128
_NUM_CORES = 6


def train():
    """ Pipeline for training data on the MNIST dataset """
    tensorboard = TensorBoard(log_dir="logs/mnist")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    training_set = data_gen(x_train, y_train, is_training=True,
                            batch_size=_BATCH_SIZE)
    testing_set = data_gen(x_test, y_test, is_training=False,
                           batch_size=_BATCH_SIZE)

    model = net()
    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    model.fit(
        training_set.make_one_shot_iterator(),
        steps_per_epoch=len(x_train) // _BATCH_SIZE,
        epochs=_EPOCHS,
        validation_data=testing_set.make_one_shot_iterator(),
        validation_steps=len(x_test) // _BATCH_SIZE,
        verbose=1,
        callbacks=[tensorboard])


def preprocess_fn(image, label):
    ''' Resize image to 32,32,1, onehot the label '''
    x = tf.reshape(tf.cast(image, tf.float32), (28, 28,1))
    x = tf.image.resize_images(x, (32, 32)) 
    y = tf.one_hot(tf.cast(label, tf.uint8), _NUM_CLASSES) 
    return x, y


def data_gen(images, labels, is_training, batch_size=128):
    """ Create a data generator """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        dataset = dataset.shuffle(1000) 

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=_NUM_CORES,
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


def net():
    """ LeNet-5 """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(6, (5, 5), input_shape=(32, 32, 1)))
    model.add(tf.keras.layers.MaxPool2D((2, 2), 2))
    model.add(tf.keras.layers.Conv2D(16, (5, 5)))
    model.add(tf.keras.layers.MaxPool2D((2, 2), 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(_NUM_CLASSES, activation='softmax'))
    return model

if __name__ == '__main__':
    train()
