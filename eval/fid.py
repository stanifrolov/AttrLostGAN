#!/usr/bin/env python3
""" Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
"""

# Adapted from https://github.com/bioinf-jku/TTUR

from __future__ import absolute_import, division, print_function

import os
import pathlib
import warnings
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import linalg


class InvalidFIDException(Exception):
    pass


def load_image_batch(files, size=None):
    """Convenience method for batch-loading images. Reduces memory requirements, at the cost of slightly reduced efficiency.
       Acknowledgement to Pyrestone.
    Params:
    -- files    : list of paths to image files. Images need to have same dimensions for all files.
    -- size     : size to which to resize the images to
    Returns:
    -- A numpy array of dimensions (num_images, height, width, 3) representing the image pixel values.
    """
    images = []
    for fn in files:
        img = Image.open(fn).convert('RGB')
        if size is not None:
            img = img.resize((size, size))
        images.append(np.asarray(img))
    images = np.array(images)

    assert images.shape[0] == len(files)
    assert images.shape[3] == 3

    return images


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    with tf.io.gfile.GFile(pth, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


# code for handling inception net derived from
# https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
                # shape = [s.value for s in shape] TF 1.x
                shape = [s for s in shape]  # TF 2.x
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
    return pool3


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_imgs, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    assert np.min(images) >= 0
    assert np.max(images) <= 256
    assert 50 <= np.mean(images) <= 200
    assert images.shape[3] == 3

    inception_layer = _get_inception_layer(sess)
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images // batch_size  # drops the last batch if < batch_size
    pred_arr = np.empty((n_batches * batch_size, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size

        if start + batch_size < n_images:
            end = start + batch_size
        else:
            end = n_images

        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch.shape[0], -1)
    if verbose:
        print(" done")
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    assert np.min(images) >= 0
    assert np.max(images) <= 256
    assert 50 <= np.mean(images) <= 200
    assert images.shape[3] == 3

    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_activations_from_files(files, size, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- size       : size to which to resize the images to
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    n_imgs = len(files)
    if batch_size > n_imgs:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_imgs
    n_batches = n_imgs // batch_size + 1
    pred_arr = np.empty((n_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        if start + batch_size < n_imgs:
            end = start + batch_size
        else:
            end = n_imgs

        batch = load_image_batch(files[start:end], size)
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size, -1)
        del batch  # clean up memory
    if verbose:
        print(" done")
    return pred_arr


def calculate_activation_statistics_from_files(files, size, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files      : list of paths to image files. Images need to have same dimensions for all files.
    -- size       : size to which to resize the images to
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations_from_files(files, size, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# -------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
# -------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    """ Checks if the path to the inception file is valid, or downloads
        the file if it is not present. """
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def _handle_path(path, num_images, size, sess, low_profile=False):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        if num_images is not None:
            assert len(files) >= num_images
            if len(files) > num_images:
                print("Found more files than required. Selecting randomly %d from %d" % (num_images, len(files)))
                import random
                random.shuffle(files)
                files = files[:num_images]
        if low_profile:
            m, s = calculate_activation_statistics_from_files(files, size, sess)
        else:
            x = load_image_batch(files, size)
            m, s = calculate_activation_statistics(x, sess)
            del x  # clean up memory
    return m, s


def calculate_fid_given_paths(paths, num_images, size, inception_path, low_profile=False):
    """ Calculates the FID of two paths. """
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], num_images, size, sess, low_profile=low_profile)
        m2, s2 = _handle_path(paths[1], num_images, size, sess, low_profile=low_profile)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("path", type=str, nargs=2,
                        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("--num_images", type=int, default=None,
                        help='Number of images to use for calculation')
    parser.add_argument("--size", type=int, required=True,
                        help='The size the loaded images should be resized to')
    parser.add_argument("-i", "--inception", type=str, default=None,
                        help='Path to Inception model (will be downloaded if not provided)')
    parser.add_argument("--lowprofile", action="store_true",
                        help='Keep only one batch of images in memory at a time. This reduces memory footprint, but may decrease speed slightly.')

    args = parser.parse_args()

    fid = calculate_fid_given_paths(args.path,
                                    args.num_images,
                                    args.size,
                                    args.inception,
                                    low_profile=args.lowprofile)
    print("FID: ", fid)
