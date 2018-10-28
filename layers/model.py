import tensorflow.contrib.slim as slim
import tensorflow as tf
from layers.net_params import default_params


def ssd_net(inputs,
            feat_layer=default_params.feat_layers,
            num_classes=default_params.num_classes,
            anchor_sizes=default_params.anchor_sizes,
            aspect_ratios=default_params.aspect_ratios,
            normalizations=default_params.normalizations,
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
        # vgg16网络
        # conv1
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')

        # 外加的ssd层
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net

        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net

        with tf.variable_scope('block8'):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1X1')
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3')
        end_points['block8'] = net

        with tf.variable_scope('block9'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3')
        end_points['block9'] = net
        with tf.variable_scope('block10'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points['block10'] = net
        with tf.variable_scope('block11'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points['block11'] = net

        # 分别获取预测和定位的feature maps
        predictions = []  # 预测类别
        logits = []  # 概率
        localizations = []  # 预测位置
        for index, layer in enumerate(feat_layer):
            with tf.variable_scope(layer + '_box'):
                cls_pred, loc_pred = multibox_layer(end_points[layer],
                                                    num_classes,
                                                    anchor_sizes[index],
                                                    aspect_ratios[index],
                                                    normalizations=normalizations[i])
                predictions.append(slim.softmax(cls_pred))
                logits.append(cls_pred)
                localizations.append(loc_pred)

        return predictions, localizations, logits, end_points


def multibox_layer(inputs,
                   num_classes,
                   anchor_sizes,
                   aspect_ratios=[1],
                   normalizations=-1):
    """
    这里默认输入格式为NHWC即[batch,height,width,channels]形式
    :param inputs:
    :param num_classes:
    :param anchor_sizes:
    :param aspect_ratios:
    :param normalizations:
    :param bn_normalization:
    :return:
    """
    net = inputs
    if normalizations > 0:
        # TODO add l2normlization
        pass
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
