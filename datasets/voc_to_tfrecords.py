import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
import os
import random
import sys

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
rootDir = 'E:/ImageFolder/VOCdevkit/VOC2007'
# 图片和标签存放的文件夹.
DIRECTORY_ANNOTATIONS = 'Annotations'
DIRECTORY_IMAGES = 'JPEGImages'
# 随机种子.

RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200  # 每个文件的样本数


# 生成整数型,浮点型和字符串型的属性
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


# 图片处理
def _process_image(directory, name):
    """
    将图片转化成二进制文件，并且别解析标注xml信息
    :param directory:
    :param name:
    :return:
    """
    class_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
    image_path = os.path.join(directory, DIRECTORY_IMAGES, name + '.jpg')
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    xml_path = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    root = ET.parse(xml_path).getroot()
    # 从标注文件中获取图片信息
    size = root.find('size')
    shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]

    # annotations
    bboxes = []
    label_indexes = []
    label_texts = []
    difficult = []
    truncated = []
    for obj in root.iter('object'):
        label = obj.find('name').text
        label_indexes.append(class_to_index[label])
        label_texts.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        ymin = float(bbox.find('ymin').text) / shape[0]
        xmin = float(bbox.find('xmin').text) / shape[1]
        ymax = float(bbox.find('ymax').text) / shape[0]
        xmax = float(bbox.find('xmax').text) / shape[1]
        if abs(ymax - ymin) < 1 and abs(xmax - xmin) < 1:
            bboxes.append([ymin, xmin, ymax, xmax])
    return image_data, shape, bboxes, label_indexes, label_texts, difficult, truncated


# 转化成tf.train.Example对象
def _convert_to_example(image_data, shape, bboxes, labels, label_texts, difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(label_texts),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


# 将example增加到tfrecord
def _add_to_tfrecord(dataset_root, file_name, tfrecod_writer):
    image_data, shape, bboxes, label_indexes, label_texts, difficult, truncated = _process_image(dataset_root,
                                                                                                 file_name)
    example = _convert_to_example(image_data, shape, bboxes, label_indexes, label_texts, difficult, truncated)
    tfrecod_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)
    pass


def run(dataset_dir=rootDir, output_dir='../tfrecords', name='voc_train', shuffling=False):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, len(filenames)))
                sys.stdout.flush()

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1
    print('\nFinished converting the Pascal VOC dataset!')


# 测试
if __name__ == '__main__':
    run()
