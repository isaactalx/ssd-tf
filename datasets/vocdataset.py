import tensorflow as tf
import tensorflow.contrib.slim as slim
from datasets.voc07_config import IMAGE_NUMBERS
import os

# 适配器1：将example 反序列化成存储之前的格式。
keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
}

# 适配器2：将反序列化的数据组装成更高级的格式。由slim完成
items_to_handlers = {
    'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
    'object/bbox': slim.tfexample_decoder.BoundingBox(
        ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
    'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
}

# 解码器
decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)


def get_dataset(record_path, num_samples, num_classes=20):
    tfr_pattren = os.path.join(record_path, '*.tfrecord')
    return slim.dataset.Dataset(
        data_sources=tfr_pattren,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions={},
        num_classes=num_classes)


def provide_data(dataset):
    # provider对象根据dataset信息读取数据
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=2,
        common_queue_capacity=20 * 5,
        common_queue_min=10 * 5,
        shuffle=True)
    image, glabels, gbboxes = provider.get(['image', 'object/label',
                                            'object/bbox'])
    return image, glabels, gbboxes


if __name__ == '__main__':
    dataset = get_dataset(record_path='../tfrecords', num_samples=IMAGE_NUMBERS['train'])

    # provider对象根据dataset信息读取数据
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * 64,
        common_queue_min=10 * 64)

    image, glabels, gbboxes = provider.get(['image',
                                            'object/label',
                                            'object/bbox'])
    pass
