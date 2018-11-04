from collections import namedtuple

SSDParams = namedtuple('SSDParameters', ['img_shape',  # 输入图片尺寸
                                         'num_classes',  # 分类数
                                         'no_annotation_label',  # 无标注标签
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',  # 默认anchor大小
                                         'aspect_ratios',  # anchor长宽比
                                         'anchor_steps',  # 特征图相对原始图像的缩放
                                         'normalizations',
                                         'prior_scaling'  # 是对特征图参考框向gtbox做回归时用到的尺度缩放（0.1,0.1,0.2,0.2）
                                         ])


def ssd300_config():
    SSDParams.img_shape = (300, 300)
    SSDParams.num_classes = 21
    SSDParams.feat_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
    SSDParams.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    SSDParams.anchor_sizes = [(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
    SSDParams.anchor_steps = [8, 16, 32, 64, 100, 300] # 特征图锚点框放大到原始图的缩放比例
    SSDParams.aspect_ratios = [[2, .5],
                               [2, .5, 3, 1. / 3],
                               [2, .5, 3, 1. / 3],
                               [2, .5, 3, 1. / 3],
                               [2, .5],
                               [2, .5]]
    SSDParams.normalizations = [20, -1, -1, -1, -1, -1]
    return SSDParams
    pass


default_params = ssd300_config()
