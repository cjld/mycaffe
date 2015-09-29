#!/usr/bin/python

import caffe
import numpy as np
from matplotlib import pylab as pl
import os
import cv2
import sys

mean = caffe.proto.caffe_pb2.BlobProto.FromString(open("./my/model/train_mean.binaryproto").read())
mn = np.array(mean.data)
mn = mn.reshape(mean.channels, mean.height, mean.width)
mn = mn.transpose((1,2,0))

files = open(sys.argv[1])

caffe.set_mode_cpu()
net = caffe.Net('./my/model/all_split/deploy.prototxt', 
                 './my/model/all_split/split_iter_93000.caffemodel', caffe.TEST)

def feednet(img):
    dsize = 1024
    overlap = 64
    zsize = dsize + overlap
    
    shape = (1,3,zsize, zsize)
    
    data_layer = net.blobs['data']
    if data_layer.data.shape == shape:
        net.blobs['data'].reshape(*shape)
        
    prob = 0
    
    for ox in range(img.shape[1]/dsize):
        for oy in range(img.shape[2]/dsize):
            offx = min(ox*dsize, img.shape[1]-zsize)
            offy = min(oy*dsize, img.shape[2]-zsize)
            cimg = img[:,offx:offx+zsize, offy:offy+zsize]
            data_layer.data[...] = cimg
            
            net.forward()
            mylabel = net.blobs['pixel-conv-tiled'].data[0]
            mylabel_sf = np.exp(mylabel) / (np.exp(mylabel[0]) + np.exp(mylabel[1]))
            prob = max(prob, mylabel_sf[1].max())
            print ox, oy, prob
            
    return prob

def crop4(full4k):
    size = full4k.shape[0]
    crop = np.roll(full4k[size/4:size/4*3, :, :], size/4, axis=1)
    crop = crop.reshape((size/2, 4, size/2, 3)).transpose((1,0,2,3))
    return crop

def transform(img):
    img = img - mn
    img = cv2.resize(img, (2048*3,2048*3), interpolation=cv2.INTER_CUBIC)
    return img.transpose(2,0,1)

names = files.readlines()
fwrite = open(sys.argv[2], 'w')

for name in names:
    name = name.strip()
    print "handle image", name
    img = cv2.imread(name)
    crop = crop4(img)
    prob = 0.0
    for i in range(4):
        rimg = transform(crop[i])
        prob = max(prob, feednet(rimg))
    fwrite.write(str(prob>=0.2)+'\n')
