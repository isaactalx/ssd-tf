import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope

from layers.net_params import default_params


def ssd_net(inputs,
            feat_layers=default_params.feat_layers,
            num_classes=default_params.num_classes,
            anchor_sizes=default_params.anchor_sizes,
            aspect_ratios=default_params.aspect_ratios,
            normalizations=default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='ssd_300_vgg'):
    """
    :param inputs:
    :param num_classes:
    :return:
    """
    # end_points:收集每一层输出内容
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', padding='SAME')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', padding='SAME')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', padding='SAME')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', padding='SAME')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4', padding='SAME')

        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', padding='SAME')
        end_points['block5'] = net

        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5', padding='SAME')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6', padding='SAME')
        end_points['block6'] = net
        # net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7', padding='SAME')
        end_points['block7'] = net
        # net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            with tf.name_scope(None, 'pad2d', [net]):
                paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
                net = tf.pad(net, paddings, mode='CONSTANT')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            with tf.name_scope(None, 'pad2d', [net]):
                paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
                net = tf.pad(net, paddings, mode='CONSTANT')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        end_point = 'block11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points[end_point] = net  # [batch,1,1,256]

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = multibox_layer(end_points[layer],
                                      num_classes,
                                      anchor_sizes[i],
                                      aspect_ratios[i],
                                      normalizations[i])
            predictions.append(slim.softmax(p))
            logits.append(p)
            localisations.append(l)
        # locations:6*[batch,fe_h,fe_w,anchor_nums,4]
        # logits:6*[batch,fe_h,fe_w,anchor_nums,4,21]
        return predictions, localisations, logits, end_points


def multibox_layer(inputs,
                   num_classes,
                   anchor_sizes,
                   aspect_ratios=[1],
                   normalizations=-1):
    """
    生成anchor，代码中似乎并没有按照论文公式来计算
    这里默认输入格式为NHWC即[batch,height,width,channels]形式
    :param inputs:feature map
    :param num_classes:类别(c+1)
    :param anchor_sizes:[(min,max)...]
    :param aspect_ratios:纵横比
    :param normalizations:是否需要l2norm
    :return:
    """
    net = inputs
    if normalizations > 0:
        # 第一个featuremap需要做一次l2正则，否则该层损失误差过大
        net = l2_normalization(net, scaling=True)
    num_anchors = len(anchor_sizes) + len(aspect_ratios)  # 两个方形+ratios个长方形
    # 位置预测
    num_loc_pred = num_anchors * 4  # 4个位置
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1] + [num_anchors, 4])
    # 类别预测
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1] + [num_anchors, num_classes])
    # loc_pred格式为[h*w*num_anchors,4]
    # cls_pred格式为[h*w*num_anchors,num_classes]
    return cls_pred, loc_pred


def tensor_shape(x, rank=3):
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def anchors_all_layers(img_shape=default_params.img_shape,
                       layers_shape=default_params.feat_shapes,
                       anchor_sizes=default_params.anchor_sizes,
                       aspect_ratios=default_params.aspect_ratios,
                       anchor_steps=default_params.anchor_steps):
    """
    计算所有feature map的所有anchor信息
    :param img_shape:
    :param layers_shape:
    :param anchor_sizes:
    :param aspect_ratios:
    :param anchor_steps:
    :return:
    """
    layer_anchors = []  # 保存每个feature map的所有anchor信息
    for i, feat_shape in enumerate(layers_shape):
        anchors = anchors_one_layer(img_shape, feat_shape, anchor_sizes[i], aspect_ratios[i], anchor_steps[i])
        layer_anchors.append(anchors)
    return layer_anchors


def anchors_one_layer(img_shape, feat_shape, sizes, ratios, step, offset=0.5, dtype=np.float32):
    """
    计算每个feature map的中心坐标,将其值归一化到原图和w,h并返回
    :param img_shape:
    :param feat_shape:
    :param sizes:min & max
    :param ratios:aspect ratios
    :param step:特征图锚点框放大到原始图的缩放比例
    :param offset:
    :return:
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]  # 对于block4[38*38],y=[[0,0,...,0],[1,1,...,1],...[37,37,...,37]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]  # 这里返回的是中心点坐标与原图的相对位置(0-1)

    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)  # h*w*1

    num_anchors = len(sizes) + len(ratios)  # 4,6,6,6,6,4

    h = np.zeros((num_anchors,), dtype=dtype)
    w = np.zeros((num_anchors,), dtype=dtype)

    # min anchor的宽高/原图
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]

    di = 1  # anchor个数偏移
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    # 返回一个feature map中每个anchor中性点，size/原图尺寸,同一个cell中心点坐标一样
    # 同一个feature map有num_anchor个不同大小的anchor
    return y, x, h, w


def l2_normalization(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    """
    conv4_3需要先进行l2正则，以减小该层和后面的误差
    """
    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        norm_dim = tf.range(inputs_rank - 1, inputs_rank)
        params_shape = inputs_shape[-1:]

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)

            outputs = tf.multiply(outputs, scale)

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
