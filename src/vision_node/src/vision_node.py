#!/usr/bin/env python

from openni import openni2
from openni import _openni2 as c_api
import ctypes
import numpy as np
import cv2
import scipy.misc
import matplotlib.pyplot as plt
import Image
import rospy
import caffe
import scipy.ndimage as ndi

openni2.initialize()

dev = openni2.Device.open_any()
mode = 'color'
#mode = 'depth'

if (mode == 'color'):
    vid_stream = dev.create_color_stream()
    vid_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = openni2.PIXEL_FORMAT_RGB888, resolutionX = 1920, resolutionY = 1080, fps = 30))
elif (mode == 'depth'):
    vid_stream = dev.create_depth_stream()
    vid_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = openni2.PIXEL_FORMAT_DEPTH_1_MM, resolutionX = 512, resolutionY = 424, fps = 30))
    

vid_stream.start()

plt.ion()

scale_factor = 3

#plt.rcParams['figure.figsize'] = [scale_factor*i for i in plt.rcParams['figure.figsize']]
plt.rcParams['figure.dpi'] = scale_factor*plt.rcParams['figure.dpi']


i = 0
while (i < 1000):
    i += 1
    frame = vid_stream.read_frame()
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
#        rect = plt.Rectangle((30,30),200,50,fill=False,
#                             edgecolor='red',linewidth=3.5)
#        ax = plt.gca()
#        ax.add_patch(rect
#    im = Image.fromarray(img)
#    im.save('/home/cmaxey/test_imgs/img%d.jpeg'%i)
#        plt.show()

    else:
        plt_img.set_data(img)
        plt.draw()
#    if i == 10:
#        print 'trying to remove'
#        rect.remove()
#    if i == 50:
#        print [method for method in dir(rect) if callable(getattr(rect,method))]
#        break


vid_stream.stop()
openni2.unload()
