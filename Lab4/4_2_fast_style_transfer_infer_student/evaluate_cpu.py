# coding:utf-8
from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy as np

BATCH_SIZE = 4
DEVICE = '/cpu:0'

os.putenv('MLU_VISIBLE_DEVICES','')

# get img_shape
def ffwd(data_in, paths_out, model, device_t='', batch_size=1):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        img_shape = get_img(data_in[0]).shape
    else:
        img_shape = data_in[0].shape

    g = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True,
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1)
    with g.as_default():
        with tf.gfile.FastGFile(model,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            input_tensor = sess.graph.get_tensor_by_name('X_content:0')
            output_tensor = sess.graph.get_tensor_by_name('add_37:0')
            batch_size = 1
            batch_shape = (batch_size, 256, 256, 3)
            num_iters = int(len(paths_out)/batch_size)
            for i in range(num_iters):
                pos = i * batch_size
                curr_batch_out = paths_out[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                if is_paths:
                    curr_batch_in = data_in[pos:pos+batch_size]
                    for j, path_in in enumerate(curr_batch_in):
                        img = get_img(path_in, [256, 256, 3])
                        X[j] = img
                else:
                    X = data_in[pos:pos+batch_size]
                start = time.time()
                _preds = sess.run(output_tensor, feed_dict={'X_content:0': X})
                end = time.time()
                for j, path_out in enumerate(curr_batch_out):
                    save_img(path_out, scipy.misc.imresize(_preds[j], img_shape))
                delta_time = end - start	
                print("Inference (CPU) processing time: %s" % delta_time)  

def ffwd_to_img(in_path, out_path, model, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, model, batch_size=1, device_t=device)

def ffwd_different_dimensions(in_path, out_path, model, 
            device_t=DEVICE, batch_size=4):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], 
            model, device_t, batch_size)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str,
                        dest='model',
                        help='dir or .pb file to load model',
                        metavar='MODEL', required=True)  

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions', 
                        help='allow different image dimensions')

    return parser

def check_opts(opts):
    exists(opts.model, 'Model not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path
        ffwd_to_img(opts.in_path, out_path, opts.model, device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.model, 
                    device_t=opts.device, batch_size=opts.batch_size)
        else :
            ffwd(full_in, full_out, opts.model, device_t=opts.device, batch_size=opts.batch_size)

if __name__ == '__main__':
    main()
