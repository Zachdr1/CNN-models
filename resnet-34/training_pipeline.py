"""
Generic pipeline for loading and transforming data, and training a network.

Adapted from
    https://gist.github.com/datlife/abfe263803691a8864b7a2d4f87c4ab8

Zach D
"""
import tensorflow as tf
import matplotlib.pyplot as plt
from resnet34 import resnet34

EPOCHS = 5
NUM_CLASSES = 10
BATCH_SIZE = 512
NUM_CORES = 6
IMAGE_SHAPE = (28, 28, 1)
TRANSFORMED_IMAGE_SHAPE = (32, 32, 1)


def training_pipeline():
    ''' Main function for the pipeline '''

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    training_set = data_gen(x_train, y_train, is_training=True,
                            batch_size=BATCH_SIZE)
    testing_set = data_gen(x_test, y_test, is_training=False,
                           batch_size=BATCH_SIZE)

    model = resnet34(input_shape=TRANSFORMED_IMAGE_SHAPE, initial_downsample=False,
                     num_classes=NUM_CLASSES)
    model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
    history = model.fit(
        training_set.make_one_shot_iterator(),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=testing_set.make_one_shot_iterator(),
        validation_steps=len(x_test) // BATCH_SIZE)

    model.save('cifar100_1.h5')
    visualize_training(history)


def data_gen(images, labels, is_training, batch_size=128):
    '''
    Create a dataset from the images and labels.
    Transform the data.
    '''
    def preprocess_fn(image, label):
        ''' Transform images '''
        x = tf.reshape(tf.cast(image, tf.float32), IMAGE_SHAPE)
        x = tf.image.resize_images(x, (TRANSFORMED_IMAGE_SHAPE[0], 
                                       TRANSFORMED_IMAGE_SHAPE[1]))
        y = tf.one_hot(tf.cast(label, tf.uint8), NUM_CLASSES)
        return x, y

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        preprocess_fn, batch_size,
        num_parallel_batches=NUM_CORES,
        drop_remainder=True if is_training else False))
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def visualize_training(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Training Visualization')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    training_pipeline()
