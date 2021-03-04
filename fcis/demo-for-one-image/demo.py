# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Haochen Zhang, Yi Li, Haozhi Qi
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fcis/cfgs/fcis_coco_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_masks import show_masks
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper
from mask.mask_transform import gpu_mask_voting, cpu_mask_voting


# get symbol
ctx_id = [int(i) for i in config.gpus.split(',')]
# pprint.pprint(config)
sym_instance = eval(config.symbol)()
sym = sym_instance.get_symbol(config, is_train=False)

# set up class names
num_classes = 81
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

target_size = config.SCALES[0][0] # 600
max_size = config.SCALES[0][1] # 1000

data_names = ['data', 'im_info']
label_names = []

max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
provide_data= [[('data', (1, 3, 600, 899)), ('im_info', (1, 3))]]
provide_label = [None]
arg_params, aux_params = load_param(cur_path + '/../model/fcis_coco', 0, process=True)
predictor = Predictor(sym, data_names, label_names,
                        context=[mx.cpu()], max_data_shapes=max_data_shape,
                        provide_data=provide_data, provide_label=provide_label,
                        arg_params=arg_params, aux_params=aux_params)


def get_detected_objects(im_name, frame, score):
    return main(im_name, frame, score)


def main(im_name, frame, score):
    data = []

    # only resize input image to target size and return scale
    im, im_scale = resize(frame, target_size, max_size, stride=config.network.IMAGE_STRIDE)

    im_tensor = transform(im, config.network.PIXEL_MEANS)
    im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
    data.append({'data': im_tensor, 'im_info': im_info})

    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]

    data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                    provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                    provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

    tic()
    scores, boxes, masks, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
    im_shapes = [data_batch.data[i][0].shape[2:4] for i in xrange(len(data_batch.data))]

    masks = masks[0][:, 1:, :, :]
    im_height = np.round(im_shapes[0][0] / scales[0]).astype('int')
    im_width = np.round(im_shapes[0][1] / scales[0]).astype('int')
    # print (im_height, im_width)
    boxes = clip_boxes(boxes[0], (im_height, im_width))
    result_masks, result_dets = cpu_mask_voting(masks, boxes, scores[0], num_classes,
                                                100, im_width, im_height,
                                                config.TEST.NMS, config.TEST.MASK_MERGE_THRESH,
                                                config.BINARY_THRESH)

    dets = [result_dets[j] for j in range(1, num_classes)]
    masks = [result_masks[j][:, 0, :, :] for j in range(1, num_classes)]
    
    print 'testing {} {:.4f}s'.format(im_name, toc())

    min_confidence = score

    # visualize
    for i in xrange(len(dets)):
        keep = np.where(dets[i][:,-1] > min_confidence)
        dets[i] = dets[i][keep]
        masks[i] = masks[i][keep]

    '''
    dets: [ numpy.ndarray([[x1 y1 x2 y2 score]]) for j in classes ]
        [x1 y1] - upper left corner of object
        [x2 y2] - lower right corner of object

    masks: [ numpy.ndarray([21, 21]) for j in classes ]
        confidence that pixel belongs to object
    '''

    for i in range(len(dets)):
        if len(dets[i]) > 0:
            for j in range(len(dets[i])):
                print('{name}: {score} ({loc})'.format(name = classes[i], score = dets[i][j][-1], loc = dets[i][j][:-1].tolist()))


if __name__ == '__main__':
    frame = cv2.imread('/root/FCIS/demo/COCO_test2015_000000000275.jpg')
    main('test', frame, 0.7)