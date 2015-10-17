#!/usr/bin/python

import caffe
import numpy as np
from matplotlib import pylab as pl
import os
import cv2
import sys
import time


def get_net():
    if len(sys.argv) >= 4:
        gid = int(sys.argv[3])
        caffe.set_mode_gpu()
        caffe.set_device(gid)
        print "Using GPU id", gid
    else:
        caffe.set_mode_cpu()
        print "Using CPU"
    net = caffe.Net('./my/model/all_split/deploy.prototxt', 
                 './my/model/all_split/split_iter_128000.caffemodel', caffe.TEST)
    return net

def feednet(img):
    dsize = 1024
    overlap = 64
    zsize_x = min(dsize + overlap, img.shape[1])
    zsize_y = min(dsize + overlap, img.shape[2])
    
    shape = (1,3,zsize_x, zsize_y)
    
    data_layer = net.blobs['data']
    if data_layer.data.shape == shape:
        net.blobs['data'].reshape(*shape)
        
    prob = 0
    
    for ox in range(img.shape[1]/dsize):
        for oy in range(img.shape[2]/dsize):
            offx = min(ox*dsize, img.shape[1]-zsize_x)
            offy = min(oy*dsize, img.shape[2]-zsize_y)
            cimg = img[:,offx:offx+zsize_x, offy:offy+zsize_y]
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

def inputer(img, oimg, name):
    global data_queue
    global cmd_queue
    dsize_x = 1024*3
    dsize_y = 1024*3
    overlap = 64
    zsize_x = min(dsize_x + overlap, img.shape[1])
    zsize_y = min(dsize_y + overlap, img.shape[2])

    cmd = img.shape[1]/dsize_x * (img.shape[2]/dsize_y)
    cmd_queue.put((cmd, name, oimg))
    
    for ox in range(img.shape[1]/dsize_x):
        for oy in range(img.shape[2]/dsize_y):
            offx = min(ox*dsize_x, img.shape[1]-zsize_x)
            offy = min(oy*dsize_y, img.shape[2]-zsize_y)
            cimg = img[:,offx:offx+zsize_x, offy:offy+zsize_y]
            data_queue.put(cimg.reshape((1,3,zsize_x,zsize_y)))

def worker():
    global result_queue
    global data_queue
    net = get_net()
    data_layer = net.blobs['data']
    result_layer = net.blobs['pixel-conv-tiled']

    last_time = time.time()
    while True:
        now_time = time.time()
        print "worker time elapsed:", now_time-last_time
        last_time = now_time

        img = data_queue.get()

        if isinstance(img, str) and img=='end':
            break
        shape = img.shape
        if data_layer.data.shape != shape:
            print "reshape data layer", shape
            data_layer.reshape(*shape)


        now_time = time.time()
        print "worker wait time elapsed:", now_time-last_time
        last_time = now_time

        data_layer.data[...] = img
        net.forward()

        now_time = time.time()
        print "worker forward time elapsed:", now_time-last_time
        last_time = now_time


        result_queue.put(np.array(result_layer.data))

def outputer():
    global cmd_queue
    global result_queue
    global fwrite
    last_time = time.time()
    while True:
        now_time = time.time()
        print "outputer time elapsed:", now_time-last_time
        last_time = now_time
        cmd, name, data = cmd_queue.get()
        if cmd == "end":
            break
        max_prob = 0
        for i in range(cmd):
            mylabel = result_queue.get()
            mylabel_sf = np.exp(mylabel) / (np.exp(mylabel[:,0:1,:,:]) + np.exp(mylabel[:,1:2,:,:]))
            prob = mylabel_sf[:,1,:,:].max()
            max_prob = max(prob, max_prob)

        print name, max_prob
        fwrite.write("{} {} {}\n".format(name,cmd,max_prob))
        fwrite.flush()

        if max_prob > 0.95:
            cv2.imwrite(name, data)


def main_serial():
    global net
    net = get_net()
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
        fwrite.flush()

def main_parallet():
    global fwrite, names, fwrite, mean, mn
    files = open(sys.argv[1])
    names = files.readlines()
    fwrite = open(sys.argv[2], 'w')
    mean = caffe.proto.caffe_pb2.BlobProto.FromString(open("./my/model/train_mean.binaryproto").read())
    mn = np.array(mean.data)
    mn = mn.reshape(mean.channels, mean.height, mean.width)
    mn = mn.transpose((1,2,0))


    import Queue
    import threading
    global cmd_queue, data_queue, result_queue
    data_queue = Queue.Queue(3*8)
    result_queue = Queue.Queue()
    cmd_queue = Queue.Queue()

    worker_t = threading.Thread(target=worker)
    outputer_t = threading.Thread(target=outputer)

    worker_t.start()
    outputer_t.start()
    print "worker start"

    output_path = os.environ['output_path']

    for name in names:
        name = name.strip()
        print "handle image", name
        img = cv2.imread(name)
        crop = crop4(img)
        basename = name.split('/')[-1]
        for i in range(4):
            rimg = transform(crop[i])
            inputer(rimg, crop[i], output_path+basename[:-4]+'.'+str(i)+basename[-4:])

    data_queue.put('end')
    cmd_queue.put(('end',0,0))

    worker_t.join()
    outputer_t.join()
    print "worker stop"

if os.environ.has_key('mode') and os.environ['mode'] == 'exclude':
    tasks = os.environ['tasks'].split(' ')
    bucket_number = int(os.environ['bucket_number'])
    bucket_size = int(os.environ['bucket_size'])
    buckets_dir = os.environ['buckets_dir']
    pnames = []
    for fname in tasks:
        names = open(fname+'/output.txt').read().splitlines()
        pwd = open(fname+'/list.txt').read().splitlines()
        dct = { ls.split('/')[-1]:ls for ls in pwd }
        for name in names:
            if float(name.split(' ')[-1]) < 0.95:
                vname = name.split(' ')[0]
                vname = vname.split('/')[-1]
                number = int(vname[-5])
                vname = vname[:-6]+vname[-4:]
                vname = dct[vname]
                pnames.append((vname,number))
    import random
    random.shuffle(pnames)
    def gen_bucket(i):
        dir_name = buckets_dir+'/bucket'+str(i)
        os.system('mkdir -p ' + dir_name)
        for name,part in pnames[i*bucket_size:(i+1)*bucket_size]:
            print i, name, part
            img = cv2.imread(name)
            crop = crop4(img)[part]
            cv2.imwrite(dir_name+'/'+name.split('/')[-1][:-3]+str(part)+name[-4:], crop)
    import threading
    ths = []
    for i in range(bucket_number):
        th = threading.Thread(target=lambda :gen_bucket(i))
        print "start thread", i
        th.start()

    for th in ths: th.join()

else:
    main_parallet()
