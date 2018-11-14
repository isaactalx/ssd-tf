import tensorflow as tf
import tensorflow.contrib.slim as slim


def ssd_losses(logits, localizations,  # 预测类别，位置
               gclasses, glocalizations, gscores,  # ground truth类别，位置，得分
               match_threshold=0.5,  # IOU阀值
               negative_ratio=3.,  # 负样本、正样本采集比
               alpha=1.,
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        logit = logits[0]
        num_classes = logit.shape[-1]
        batch_size = logit.shape[0].value

        flogits = []
        fgclasses = []
        fgscores = []
        flocalizations = []
        fglocalizations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalizations.append(tf.reshape(localizations[i], [-1, 4]))
            fglocalizations.append(tf.reshape(glocalizations[i], [-1, 4]))
        logits = tf.concat(flogits, axis=0)  # 全部搜索框，对应21个类别的输出[8732*batch_size,21]
        gclasses = tf.concat(fgclasses, axis=0)  # 全部搜索框,真实类别的数字[8732*batch_size,]
        gscores = tf.concat(fgscores, axis=0)  # 全部搜索框和真实框的IOU[8732*batch_size,]
        localizations = tf.concat(flocalizations, axis=0)  # [8732*batch_size,4]
        glocalizations = tf.concat(fglocalizations, axis=0)  # [8732*batch_size,4]

        dtype = logits.dtype

        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)  # 浮点型前景掩码
        n_positives = tf.reduce_sum(fpmask)  # 前景总数

        # 难分样本挖掘
        # 1.根据正样本个数确定负样本个数n，并找出confidece loss最大的n个负样本，计算分类损失之和
        # 2.计算整合样本分类损失之和、负样本分类损失之和
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)  # 每一行的分类预测转化为概率
        nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)  # IOU达不到阀值的类别搜索框记为1
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask, predictions[:, 0],  # 框内无物体标记为背景预测概率
                           1. - fnmask)  # 框内有物体位置标记为1
        nvalues_flat = tf.reshape(nvalues, [-1])

        # 在nmask中剔除n_neg个最不可能为背景的框
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        # 3*前景总数+batch_size
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)
        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  # 最不可能为背景的n_neg个点
        max_hard_pred = -val[-1]
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # 不是前景，又最不像背景的n_neg个点
        fnmask = tf.cast(nmask, dtype)

        # 计算正负样本分类损失
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)  # 0-20
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)  # {0,1}
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # 计算定位损失
        with tf.name_scope('localization'):
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = abs_smooth(localizations - glocalizations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)


def abs_smooth(x):
    """
    smooth L1函数
    :return:1/2*x^2 if |x|<1 else |x|-1/2
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r
