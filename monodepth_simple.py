# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import cv2
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

import matplotlib.cm
cmap = matplotlib.cm.get_cmap("viridis")

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--image_path',       type=str,   help='path to the image', required=True)
parser.add_argument('--output_path',       type=str,   help='path to the image', required=True)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)
parser.add_argument('--stereo_baseline',  type=float, help='stereo baseline used in the training dataset (in meters)', default=0.472)

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    # Note: This panoramic implementation doesn't use the blending between left/right and middle disparity used in the original monocular depth paper
    #       because this weighted blending introduces discontinuities at the edges that are undesirable for panoramic images.
    return m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None, use_ring_pad=True)

    if os.path.isdir(args.image_path):
        image_list = sorted(os.listdir(args.image_path))
        output_directory = args.output_path
        is_dir = True
    else:
        image_list = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
        is_dir = False

    stereo_baseline = args.stereo_baseline

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = args.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    for image_path in image_list:
        print(image_path)
        if is_dir:
            image_path = os.path.join(args.image_path, image_path)
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = os.path.join(output_directory, "{}.npy".format(output_name))

        input_image = scipy.misc.imread(image_path, mode="RGB")
        original_height, original_width, num_channels = input_image.shape

        minor_height = int(args.input_width/input_image.shape[1] * input_image.shape[0]/0.7191)
        input_image = scipy.misc.imresize(input_image, [minor_height, args.input_width], interp='lanczos')

        input_image = np.pad(input_image, ((args.input_height-minor_height,0),(0,0),(0,0)), mode='constant')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

        disp_pp = disp_pp[args.input_height-minor_height:]


        pred_width = disp_pp.shape[1]
        disp_pp = cv2.resize(disp_pp.squeeze(), (original_width, original_height))

        angular_precision = original_width/math.pi
        disp_pp *= angular_precision/stereo_baseline*original_width/pred_width
        np.save(output_filename, disp_pp)

        plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_pp, cmap='viridis')

    print('done!')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False,
        equirectangular_mode=True,
        fov=0) # FOV is not needed during testing time.

    test_simple(params)

if __name__ == '__main__':
    tf.app.run()
