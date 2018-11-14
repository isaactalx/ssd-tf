from datasets import dataset
from datasets.voc07_config import IMAGE_NUMBERS
import tensorflow as tf
import tensorflow.contrib.slim as slim
from layers import net_builder
from preprocess import preprocess_for_train
from layers.net_params import default_params
from layers.bbox_methods import bboxes_encode
from layers.ssd_net import SSDNet
import os
import time

max_steps = 1500
batch_size = 32
adam_beta1 = 0.9
adam_beta2 = 0.999
opt_epsilon = 1.0
num_epochs_per_decay = 2.0
num_samples_per_epoch = 17125


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


def gen_data_queue():
    datasets = dataset.get_dataset(record_path='./tfrecords', num_samples=IMAGE_NUMBERS['train'])
    # provider对象根据dataset信息读取数据
    ssd_anchors = net_builder.anchors_all_layers()
    with tf.device('/device:CPU:0'):
        with tf.name_scope('voc_2007_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                datasets,
                num_readers=4,
                common_queue_capacity=20 * 64,
                common_queue_min=10 * 64,
                shuffle=True)
        # Get for SSD network: image, labels, bboxes.
        [image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
                                                         'object/label',
                                                         'object/bbox'])
        # 图片预处理
        image, glabels, gbboxes = \
            preprocess_for_train(image, glabels, gbboxes,
                                 default_params.img_shape)
        # groundtruth box编码
        gclasses, glocalisations, gscores = \
            bboxes_encode(glabels, gbboxes, ssd_anchors)

        batch_shape = [1] + [len(ssd_anchors)] * 3

        r = tf.train.batch(
            reshape_list([image, gclasses, glocalisations, gscores]),
            batch_size=32,
            num_threads=4,
            capacity=5 * 32)
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            reshape_list(r, batch_shape)

        batch_queue = slim.prefetch_queue.prefetch_queue(
            reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
            capacity=2)
        return batch_queue, batch_shape


def train():
    with tf.Graph().as_default():
        # Create global_step.
        with tf.device("/device:CPU:0"):
            global_step = tf.train.create_global_step()

        ssd = SSDNet()
        batch_queue, batch_shape = gen_data_queue()

        b_image, b_gclasses, b_glocalisations, b_gscores = \
            reshape_list(batch_queue.dequeue(), batch_shape)  # 重整list
        predictions, localisations, logits, end_points = \
            ssd.net(b_image, is_training=True, weight_decay=0.00004)
        ssd.losses(logits, localisations,
                   b_gclasses, b_glocalisations, b_gscores)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.device("/device:CPU:0"):  # learning_rate节点使用CPU（不明）
            decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
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
            tf.summary.scalar('learning_rate', learning_rate)

            trainable_scopes = None
            if trainable_scopes is None:
                variables_to_train = tf.trainable_variables()
            else:
                scopes = [scope.strip() for scope in trainable_scopes.split(',')]
                variables_to_train = []
                for scope in scopes:
                    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                    variables_to_train.extend(variables)

            losses = tf.get_collection(tf.GraphKeys.LOSSES)
            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses)
            loss = tf.add_n(losses)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("regularization_loss", regularization_loss)

            grad = optimizer.compute_gradients(loss, var_list=variables_to_train)
            grad_updates = optimizer.apply_gradients(grad, global_step=global_step)

            update_ops.append(grad_updates)

            with tf.control_dependencies(update_ops):
                total_loss = tf.add_n([loss, regularization_loss])
            tf.summary.scalar("total_loss", total_loss)

            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            config = tf.ConfigProto(log_device_placement=False,
                                    gpu_options=gpu_options)
            saver = tf.train.Saver(max_to_keep=5,
                                   keep_checkpoint_every_n_hours=1.0,
                                   write_version=2,
                                   pad_step_number=False)

            model_path = './logs'
            with tf.Session(config=config) as sess:
                summary = tf.summary.merge_all()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                writer = tf.summary.FileWriter(model_path, sess.graph)

                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                init_op.run()
                # begin to train
                for step in range(max_steps):
                    start_time = time.time()
                    loss_value = sess.run(total_loss)

                    duration = time.time() - start_time
                    if step % 10 == 0:
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)

                        examples_per_sec = batch_size / duration
                        sec_per_batch = float(duration)
                        format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
                        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                    if step % 500 == 0 and step != 0:
                        saver.save(sess, os.path.join(model_path, "ssd_tf.model"), global_step=step)
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    train()
