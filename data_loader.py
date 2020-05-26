import tensorflow as tf
import config
import pathlib


def get_all(data_dir):
    data_root = pathlib.Path(data_dir)
    all_images_path = [str(path) for path in list(data_root.glob('*/*'))]
    labels_name = sorted(item.name for item in data_root.glob('*/'))
    label_to_index = dict((index, label) for label, index in enumerate(labels_name))
    all_images_label = [label_to_index[pathlib.Path(single_path).parent.name] for single_path in all_images_path]

    return all_images_path, all_images_label


def process_data(path, label):
    image_raw = tf.io.read_file(path)
    image_tensor = tf.image.decode_jpeg(image_raw, channels=config.image_channels)
    image_tensor = tf.image.resize(image_tensor, [config.image_size, config.image_size])
    image_tensor = tf.cast(image_tensor, dtype=tf.float32) / 255.
    return image_tensor, label


def get_data(data_dir):
    images, labels = get_all(data_dir)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds, len(images)


def build_dataset():
    train_ds, train_len = get_data(config.train_dir)
    valid_ds, valid_len = get_data(config.valid_dir)
    test_ds, test_len = get_data(config.test_dir)

    train_ds = train_ds.shuffle(buffer_size=train_len).batch(config.batch_size)
    test_ds = test_ds.batch(config.batch_size)
    valid_ds = valid_ds.batch(config.batch_size)

    return train_ds, train_len, test_ds, test_len, valid_ds, valid_len
