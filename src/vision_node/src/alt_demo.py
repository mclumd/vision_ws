#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import ctypes
from openni import openni2
from openni import _openni2 as c_api
import scipy.ndimage as ndi

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    rects = []
#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='red', linewidth=3.5)

        rects.append(rect)
        ax.add_patch(rect)
#        ax.text(bbox[0], bbox[1] - 2,
#                '{:s} {:.3f}'.format(class_name, score),
#                bbox=dict(facecolor='blue', alpha=0.5),
#                fontsize=14, color='white')

#    ax.set_title(('{} detections with '
#                  'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                  thresh),
#                  fontsize=14)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.draw()
    return rects

def kinect_vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, im, ax, old_rects):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
 #   im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
 #   im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
 #   timer = Timer()
 #   timer.tic()
    for r in old_rects:
        r.remove()
    scores, boxes = im_detect(net, im)
#    timer.toc()
#    print ('Detection took {:.3f}s for '
#           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

#    print 'boxes shape: %s'%str(boxes.shape)
#    print 'scores shape: %s'%str(scores.shape)
    # Visualize detections for each class
    total_rects = []
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
#        print cls_boxes.shape
#        print cls_scores.shape
#        print cls_scores[:,np.newaxis].shape
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
#        print dets.shape
        keep = nms(dets, NMS_THRESH)
#        print len(keep)
        dets = dets[keep, :]
#        print dets.shape
#        print ''
        rects = vis_detections(im, ax, cls, dets, thresh=CONF_THRESH)
        for r in rects:
            total_rects.append(r)
    return total_rects

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    openni2.initialize()

    dev = openni2.Device.open_any()
    mode = 'color'
    #mode = 'depth'                                                                                          
    
    if (mode == 'color'):
        vid_stream = dev.create_color_stream()
        vid_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = openni2.PIXEL_FORMAT_RGB888, resolutionX= 1920, resolutionY = 1080, fps = 30))
    elif (mode == 'depth'):
        vid_stream = dev.create_depth_stream()
        vid_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM, resolutionX = 512, resolutionY = 424, fps = 30))
            

    vid_stream.start()
            
    plt.ion()

    scale_factor = 3
        
    #plt.rcParams['figure.figsize'] = [scale_factor*i for i in plt.rcParams['figure.figsize']]               
    plt.rcParams['figure.dpi'] = scale_factor*plt.rcParams['figure.dpi']
        
    rects = []
    i = 0
    while (True):
        i += 1
        frame = vid_stream.read_frame()
        print 'MADE IT!'
        if (mode == 'color'):
            frame_data = frame.get_buffer_as_uint8()
            img = np.frombuffer(frame_data,dtype=np.uint8)
            img.shape = (1080,1920,3)
            #        new_img = img[:,60:1860,:]                                                                      
            #        print new_img.shape                                                                             
            new_img = img[165:915:2,460:1460:2,:]
            img = new_img
            #        new_new_img = ndi.interpolation.zoom(img[:,240:1680,:],(0.347,0.347,1))                         
            #        print new_img.shape                                                                             
            
        elif (mode == 'depth'):
            frame_data = frame.get_buffer_as_uint16()
            img = np.frombuffer(frame_data,dtype=np.uint16)
            img.shape = (424,512,1)
            img = np.concatenate((img,img,img),axis=2)
        if not i > 1:
            plt_img = plt.imshow(img,aspect='equal',vmin=0, vmax=256)
#            rect = plt.Rectangle((30,30),200,50,fill=False,
#                                 edgecolor='red',linewidth=3.5)
            ax = plt.gca()
            rects = demo(net, img, ax, rects)
#            ax.add_patch(rect)
            plt.show()
        else:
            plt_img.set_data(img)
            rects = demo(net, img, ax, rects)
            plt.draw()
            



    #plt.show()
    vid_stream.stop()
    openni2.unload()
