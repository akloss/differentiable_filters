"""
Utility functions to make code work with tensorflow 1 and 2.
"""

import tensorflow as tf
import tensorflow_probability as tfp

def get_dim_int(tensor, dim):
    """
    Compatability function to get an int shape value for dimension dim of a
    tensor.

    Parameters
    ----------
    tensor : tf.tensor
        The input tensor
    dim : int
        The Dimension who's shape we want to know'

    Returns
    -------
    shape : int
        The shape of dimension dim

    """

    try:
        # this should work only in tensorflow 1
        shape = tensor.get_shape()[dim].value
    except AttributeError:
        # this should work in tensorflow 2
        shape = tensor.get_shape()[dim]
    return shape


def fill_triangular(tensor):
    """
    Compatability function to create a triangular matrix from a flat tensor.

    Parameters
    ----------
    tensor : tensor
        The flat input tensor

    Raises
    ------
    Exception
        If no implementation of "fill_triangular" was found.

    Returns
    -------
    out : tensor
        The triangular matrix.

    """
    try:
        # this works for higher versions of tensorflow probability
        out = tfp.math.fill_triangular(tensor)
    except AttributeError:
        # this works on tensorflow 1.14 with tensorflow-probability==0.7.0
        out = tf.contrib.distributions.fill_triangular(tensor)
    except:
        raise Exception('No working implementation found for method ' +
                        '\"fill_triangular\". Tried tfp.math.fill_triangular ' +
                        'and tf.contrib.distributions.fill_triangular.')
    return out