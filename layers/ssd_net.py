from layers.net_params import default_params
import tensorflow.contrib.slim as slim
import layers.net_builder as net_builder
import tensorflow as tf
from layers.loss_function import ssd_losses
import numpy as np


def ssd_losses_tmp(logits, localisations,  # 预测类别，位置
               gclasses, glocalisations, gscores,  # ground truth类别，位置，得分
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        # 提取类别数和batch_size
        lshape = tensor_shape(logits[0], 5)  # tensor_shape函数可以取代
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):  # 按照图片循环
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)  # 全部的搜索框，对应的21类别的输出
        gclasses = tf.concat(fgclasses, axis=0)  # 全部的搜索框，真实的类别数字
        gscores = tf.concat(fgscores, axis=0)  # 全部的搜索框，和真实框的IOU
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)

        dtype = logits.dtype
        pmask = gscores > match_threshold  # (全部搜索框数目, 21)，类别搜索框和真实框IOU大于阈值
        fpmask = tf.cast(pmask, dtype)  # 浮点型前景掩码（前景假定为含有对象的IOU足够的搜索框标号）
        n_positives = tf.reduce_sum(fpmask)  # 前景总数

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)  # 此时每一行的21个数转化为概率
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)  # IOU达不到阈值的类别搜索框位置记1
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],  # 框内无物体标记为背景预测概率
                           1. - fnmask)  # 框内有物体位置标记为1
        nvalues_flat = tf.reshape(nvalues, [-1])

        # Number of negative entries to select.
        # 在nmask中剔除n_neg个最不可能背景点(对应的class0概率最低)
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        # 3 × 前景掩码数量 + batch_size
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  # 最不可能为背景的n_neg个点
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # 不是前景，又最不像背景的n_neg个点
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)  # 0-20
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)  # {0,1}
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        # get_shape返回值，with_rank相当于断言assert，是否rank为指定值
        static_shape = x.get_shape().with_rank(rank).as_list()
        # tf.shape返回张量，其中num解释为"The length of the dimension `axis`."，axis默认为0
        dynamic_shape = tf.unstack(tf.shape(x), num=rank)
        # list，有定义的给数字，没有的给tensor
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


class SSDNet(object):

    def __init__(self):
        self.params = default_params

    def net(self, input_data, weight_decay, update_feat_shapes=True, is_training=True):
        with slim.arg_scope(self._ssd_arg_scope(weight_decay)):
            output = net_builder.ssd_net(input_data, is_training=is_training)
            # Update feature shapes (try at least!)
        if update_feat_shapes:
            feat_shapes = []
            # 获取各个中间层shape（不含0维），如果含有None则返回默认的feat_shapes
            for l in output[0]:
                if isinstance(l, np.ndarray):
                    shape = l.shape
                else:
                    shape = l.get_shape().as_list()
                shape = shape[1:4]
                if None in shape:
                    feat_shapes = self.params.feat_shapes
                    break
                else:
                    feat_shapes.append(shape)
            self.params = self.params._replace(feat_shapes=feat_shapes)
        return output

    def _ssd_arg_scope(self, weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME') as sc:
                return sc

    @staticmethod
    def losses(logits, localizations,  # 预测类别，位置
               gclasses, glocalizations, gscores,  # ground truth类别，位置，得分
               match_threshold=0.5,  # IOU阀值
               negative_ratio=3.,  # 负样本、正样本采集比
               alpha=1.,
               scope='ssd_losses'):
        return ssd_losses(logits, localizations,
                          gclasses, glocalizations, gscores,
                          match_threshold,
                          negative_ratio,
                          alpha,
                          scope)

    @property
    def anchors(self):
        return net_builder.anchors_all_layers()
