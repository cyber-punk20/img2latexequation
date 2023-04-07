from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys

from preprocessing import *
from pix2equation import *


def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    