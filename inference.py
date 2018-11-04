import argparse

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import layers.bbox_methods as bbox_methods
import visualization
from layers import net_builder

# init arguments
parser = argparse.ArgumentParser(description='SSD inference with tensorflow')
parser.add_argument('--image', type=str, default='./images/dog.jpg', help='file path for an image.')

args = parser.parse_args()


def process_image(predictions, localizations, nms_threshold=.45):
    rbbox_img = np.array([0., 0., 1., 1.])
    ssd_anchors = net_builder.anchors_all_layers()  # 为每一层feature map生成默认框
    rclasses, rscores, rbboxes = bbox_methods.ssd_bboxes_select(predictions, localizations, ssd_anchors)
    rbboxes = bbox_methods.bboxes_clip(rbbox_img, rbboxes)#边界狂裁剪
    #top_k+nms过滤边界框
    rclasses, rscores, rbboxes = bbox_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = bbox_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    rbboxes = bbox_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def inference(input_img):
    resized_img = cv2.resize(rgb_image, (300, 300))
    x = resized_img.astype(np.float32)
    x -= (123., 117., 104)
    x = x.astype(np.float32)
    imgs = x[:, :, ::-1].copy()

    images = np.array([imgs])
    with tf.Graph().as_default():
        inputs = tf.placeholder(tf.float32, [1, 300, 300, 3])
        predictions, localizations, logits, end_points = net_builder.ssd_net(inputs, is_training=False)

        with tf.Session() as sess:
            variables_to_restore = slim.get_variables_to_restore()
            init_fn = slim.assign_from_checkpoint_fn('./checkpoints/ssd_300_vgg.ckpt', variables_to_restore)
            init_fn(sess)
            pred, loc, log = sess.run([predictions, localizations, logits],
                                      feed_dict={inputs: images})
            rclasses, rscores, rbboxes = process_image(pred, loc)
            # draw boxes and labels
            visualization.plt_bboxes(input_img, rclasses, rscores, rbboxes)


if __name__ == '__main__':
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)  # 读出来是BGR
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inference(rgb_image)
