from layers.net_params import default_params
import tensorflow.contrib.slim as slim
import layers.net_builder as net_builder
import tensorflow as tf
from layers.loss_function import ssd_losses
import numpy as np




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

    @staticmethod
    def _ssd_arg_scope(weight_decay=0.0005):
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
