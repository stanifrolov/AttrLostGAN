# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A library to evaluate Inception on a single GPU.
"""

# Adapted from https://github.com/openai/improved-gan

import os
import math
import os.path
import sys
import tarfile

import numpy as np

print("Disabled info and warning logs")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from PIL import Image
from six.moves import urllib
from tqdm import tqdm

tf.app.flags.DEFINE_string('image_folder', None,
                           "The path where to load the images")
tf.app.flags.DEFINE_integer('num_images', 30000,
                            'The number of images to be used')
tf.app.flags.DEFINE_integer('splits', 10,
                            "The number of splits")
tf.app.flags.DEFINE_integer('gpu', 0,
                            "The ID of GPU to use")

FLAGS = tf.app.flags.FLAGS
# Batch normalization. Constant governing the exponential moving average of
# the 'global' mean and variance for all activations.
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

MODEL_DIR = '/tmp'
if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=FLAGS.splits):
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 1
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in tqdm(range(n_batches)):
            # sys.stdout.write(".")
            # sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'InputTensor:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


def load_data(path_to_images):
    print(path_to_images)
    images = []
    for path, subdirs, files in os.walk(path_to_images):
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    img = Image.open(filename).convert('RGB')
                    images.append(np.asarray(img))
    assert len(images) >= FLAGS.num_images

    if len(images) > FLAGS.num_images:
        print("Found %d images but need only %d. Selecting randomly." % (len(images), FLAGS.num_images))
        import random
        images = random.sample(images, FLAGS.num_images)
    print('images', len(images), images[0].size)
    return images


# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        print()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Import model with a modification in the input tensor to accept arbitrary
        # batch size.
        input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3],
                                      name='InputTensor')
        _ = tf.import_graph_def(graph_def, name='',
                                input_map={'ExpandDims:0': input_tensor})
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        #w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        #logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        #softmax = tf.nn.softmax(logits)
        softmax = sess.graph.get_tensor_by_name('softmax:0')
        return softmax


softmax = _init_inception()


def main(unused_argv=None):
    images = load_data(FLAGS.image_folder)
    mean, std = get_inception_score(images)
    print("IS mean: %.3f, std: %.3f" % (mean, std))


if __name__ == '__main__':
    tf.app.run()
