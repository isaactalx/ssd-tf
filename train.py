import argparse
import os
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets import vocdataset
from datasets.voc07_config import IMAGE_NUMBERS
from layers.bbox_methods import bboxes_encode
from layers.net_params import default_params
from layers.ssd_net import SSDNet
from preprocess import preprocess_for_train

adam_beta1 = 0.9
adam_beta2 = 0.999
opt_epsilon = 1.0
num_epochs_per_decay = 2.0

# init arguments
parser = argparse.ArgumentParser(description='SSD train with tensorflow')
parser.add_argument('--batch_size', type=int, default=16, help='image numbers per batch.')
parser.add_argument('--max_steps', type=int, default=1500, help='max training iterations.')

args = parser.parse_args()


def reshape_list(l, shape=None):
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i + s])
            i += s
    return r


def gen_data_queue(ssd_anchors):
    datasets = vocdataset.get_dataset(record_path='./tfrecords', num_samples=IMAGE_NUMBERS['train'])
    # provider对象根据dataset信息读取数据

    with tf.device('/device:CPU:0'):
        image, glabels, gbboxes = vocdataset.provide_data(datasets)
        # 图片预处理
        image, glabels, gbboxes = \
            preprocess_for_train(image, glabels, gbboxes,
                                 default_params.img_shape)
        # groundtruth box编码
        gclasses, glocalizations, gscores = \
            bboxes_encode(glabels, gbboxes, ssd_anchors)

        r = tf.train.batch(
            reshape_list([image, gclasses, glocalizations, gscores]),
            batch_size=args.batch_size,
            num_threads=4,
            capacity=5 * args.batch_size)

        batch_queue = slim.prefetch_queue.prefetch_queue(r, capacity=2)
        return batch_queue


def train():
    with tf.Graph().as_default():

        ssd = SSDNet()
        ssd_anchors = ssd.anchors

        with tf.device('/device:CPU:0'):
            global_step = tf.train.create_global_step()

        batch_queue = gen_data_queue(ssd_anchors)

        batch_shape = [1] + [len(ssd_anchors)] * 3
        b_image, b_gclasses, b_glocalizations, b_gscores = \
            reshape_list(batch_queue.dequeue(), batch_shape)  # 重整list
        predictions, localizations, logits, end_points = \
            ssd.net(b_image, is_training=True, weight_decay=0.00004)
        ssd.losses(logits, localizations,
                   b_gclasses, b_glocalizations, b_gscores)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.device('/device:CPU:0'):
            decay_steps = int(IMAGE_NUMBERS['train'] / args.batch_size * num_epochs_per_decay)
            learning_rate = tf.train.exponential_decay(0.01,
                                                       global_step,
                                                       decay_steps,
                                                       0.94,  # learning_rate_decay_factor,
                                                       staircase=True,
                                                       name='exponential_decay_learning_rate')
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=opt_epsilon)

            variables_to_train = tf.trainable_variables()
            losses = tf.get_collection(tf.GraphKeys.LOSSES)
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses)
            loss = tf.add_n(losses)
            #
            grad = optimizer.compute_gradients(loss, var_list=variables_to_train)
            grad_updates = optimizer.apply_gradients(grad, global_step=global_step)

            update_ops.append(grad_updates)

            with tf.control_dependencies(update_ops):
                total_loss = tf.add_n([loss, regularization_loss])

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            config = tf.ConfigProto(log_device_placement=False,
                                    gpu_options=gpu_options)
            saver = tf.train.Saver(max_to_keep=5,
                                   keep_checkpoint_every_n_hours=1.0,
                                   write_version=2,
                                   pad_step_number=False)

            model_path = './logs'
            with tf.Session(config=config) as sess:

                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                init_op.run()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # begin to train
                for step in range(args.max_steps):
                    start_time = time.time()
                    loss_value = sess.run(total_loss)
                    duration = time.time() - start_time
                    if step % 10 == 0:
                        examples_per_sec = args.batch_size / duration
                        sec_per_batch = float(duration)
                        format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
                        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                    if step % 500 == 0 and step != 0:
                        saver.save(sess, os.path.join(model_path, "ssd_tf.model"), global_step=step)
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    train()
