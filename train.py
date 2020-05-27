import os
import tensorflow as tf
from model import inception_v3
from data_loader import build_dataset
import config
import datetime
import math
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load model
model = inception_v3.InceptionV3(num_class=config.classes)
model.build(input_shape=(None, config.image_size, config.image_size, config.image_channels))

# Load dataset
train_ds, train_len, test_ds, test_len, valid_ds, valid_len = build_dataset()

# Loss and optimizer
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizers = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_acc')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_acc = tf.keras.metrics.SparseCategoricalCrossentropy(name='valid_acc')

# checkpoint
checkpoints_dir = 'checkpoints/'
checkpoint = tf.train.Checkpoint(model=model, optimizers=optimizers)
checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoints_dir, max_to_keep=1)
status = checkpoint.restore(checkpoint_manager.latest_checkpoint)

# save log
current_time = datetime.datetime.now().strftime('%Y%m%d-H%M%S')
train_log_dir = os.path.join('logs/gradient_tape', current_time, 'train')
test_log_dir = os.path.join('logs/gradient_tape', current_time, 'test')

train_summary = tf.summary.create_file_writer(train_log_dir)
test_summary = tf.summary.create_file_writer(test_log_dir)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss_aux = loss_object(y_true=labels, y_pred=predictions.aux_logits)
        loss = 0.5 * loss_aux + 0.5 * loss_object(y_true=labels, y_pred=predictions.logits)

    gradient = tape.gradient(loss, model.trainable_variables)
    optimizers.apply_gradients(zip(gradient, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions.logits)


@tf.function
def valid_step(images, labels):
    predictions = model(images, include_aux_logits=False, training=False)
    loss = loss_object(labels, predictions)

    valid_loss(loss)
    valid_acc(labels, predictions)


def main():
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    for epoch in range(config.epochs):
        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()

        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)
            template = 'Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}'
            print(template.format(epoch + 1,
                                  config.epochs,
                                  step,
                                  math.ceil(train_len / config.batch_size),
                                  train_loss.result(),
                                  train_acc.result()))
        for images, labels in valid_ds:
            valid_step(images, labels)

        template = 'Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, ' \
                   'valid loss: {:.5f}, valid accuracy: {:.5f}'
        print(template.format(epoch + 1,
                              config.epochs,
                              train_loss.result(),
                              train_acc.result(),
                              valid_loss.result(),
                              valid_acc.result()))

        with train_summary.as_default():
            tf.summary.scalar('train loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train acc', train_acc.result(), step=epoch)
        with test_summary.as_default():
            tf.summary.scalar('valid loss', valid_loss.result(), step=epoch)
            tf.summary.scalar('valid acc', valid_acc.result(), step=epoch)

        checkpoint_manager.save()

        train_acc_history.append(train_acc.result())
        train_loss_history.append(train_loss.result())
        valid_acc_history.append(valid_acc.result())
        valid_loss_history.append(valid_loss.result())

    plt.plot(train_acc_history)
    plt.plot(valid_acc_history)
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(train_loss_history)
    plt.plot(valid_loss_history)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
