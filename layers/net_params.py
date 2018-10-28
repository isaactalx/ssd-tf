from collections import namedtuple

SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_size_bounds',
                                         'anchor_sizes',
                                         'aspect_ratios',
                                         'anchor_steps',
                                         'anchor_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])




def ssd300_config():
    SSDParams.img_shape = (300, 300)
    SSDParams.num_classes = 21
    SSDParams.feat_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
    SSDParams.feat_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
    SSDParams.anchor_sizes = [(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
    SSDParams.aspect_ratios = [[2, .5],
                               [2, .5, 3, 1. / 3],
                               [2, .5, 3, 1. / 3],
                               [2, .5, 3, 1. / 3],
                               [2, .5],
                               [2, .5]]
    SSDParams.normalizations=[20, -1, -1, -1, -1, -1]
    return SSDParams
    pass

default_params=ssd300_config()