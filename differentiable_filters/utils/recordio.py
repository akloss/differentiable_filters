"""
Tools to write data into tf.TFRecords and read data from tf.TFRecords.
Adapted from code by Daniel Kappler
"""

import types
import collections
import os
import shutil
import yaml

import numpy as np
import tensorflow as tf


class RecordioWriter(object):
    def __init__(self, dir_path_records, num_examples_per_record, prefix=''):
        """
        Class to write out tfrecord files from data. If necessary, the data is
        split into multiple tfrecord files with a fixed maximum size.
        We store the data in *.tfrecord files where the first number is the
        record index and the second one the total number of records.

        Parameters
        ----------
        dir_path_records : str
            Output directory.
        num_examples_per_record : int
            Maximum number of tf.Examples allowed in one file
        prefix : str, optional
            Optional name prefix for the files. The default is ''.

        Returns
        -------
        None.

        """
        self._dir_path_records = dir_path_records
        self._num_examples_per_record = num_examples_per_record
        self._prefix = prefix
        self._file_path_records = []
        self._num_examples = 0
        self._writer = None

    @property
    def num_records(self):
        return len(self._file_path_records)

    @property
    def num_examples(self):
        return self._num_examples

    def close(self):
        if self._writer is not None:
            self._writer.close()
        self._update_file_path_records()

    def write(self, serialized_example_proto):
        if self._num_examples % self._num_examples_per_record == 0:
            self._update_writer()
        self._writer.write(serialized_example_proto)
        self._num_examples += 1

    def _update_writer(self):
        file_path_record = \
            os.path.join(self._dir_path_records,
                         self._prefix +
                         self._get_record_file_name(self.num_records, 0))

        self._file_path_records.append(file_path_record)
        self._writer = tf.io.TFRecordWriter(file_path_record)

    def _update_file_path_records(self):
        for pos, file_path_record in enumerate(self._file_path_records):
            shutil.move(file_path_record,
                        os.path.join(self._dir_path_records,
                                     self._prefix +
                                     self._get_record_file_name(pos,
                                                                self.num_records)))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _get_record_file_name(self, record, num_records=None):
        if num_records is not None:
            return '{0:05d}_{1:05d}.tfrecord'.format(record, num_records)
        else:
            return '{0:05d}.tfrecord'.format(record)


class RecordMeta(object):
    def __init__(self, prefix=''):
        """
        Class that stores and writes out meta information about the tfrecords
        written by a RecordioWriter

        Parameters
        ----------
        prefix : str, optional
            Optional name prefix for the meta files. The default is ''.

        Returns
        -------
        None.

        """
        self.num_examples = 0
        self.num_records = 0
        self.prefix = prefix
        self.tensors_dtype = {}
        self.tensors_shape = {}
        self.tensors_shape_max = {}
        self.tensors_ndim = {}

    def save(self, dir_path_records):
        # check
        file_path = os.path.join(dir_path_records, self.prefix + 'meta.yaml')
        data = {}
        variables = vars(self)
        for key, value in variables.items():
            if type(value) == dict:
                for k, v in value.items():
                    if type(v) == map:
                        value[k] = list(v)
            if key[0] != '_':
                data[key] = value
        # save(file_path, data)
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, 'w') as fo:
            yaml.dump(data, fo)

    def keys(self):
        return self.tensors_shape.keys()

    def add_tf_feature(self, key, features):
        tensor_shape = self.tensors_shape[key]
        dtype_load = _get_tf_dtype(self.tensors_dtype[key])
        if self._is_static_shape(key):
            features[key + '/data'] = tf.io.FixedLenFeature(
                dtype=dtype_load, shape=tensor_shape)
        else:
            features[key + '/data'] = tf.io.VarLenFeature(dtype=dtype_load)
            features[key + '/dims'] = tf.io.FixedLenFeature(
                shape=(self.tensors_ndim[key],),
                dtype=tf.int64)

    def reshape_and_cast(self, key, tensors):
        with tf.name_scope('reshape_and_cast/{}'.format(key)):
            dtype_run = _get_tf_dtype(self.tensors_dtype[key])
            tensor = tensors[key + '/data']

            if not self._is_static_shape(key):
                runtime_shape = tensors[key + '/dims']
                static_shape = self.tensors_shape[key]
                shape = []
                for pos, static in enumerate(static_shape):
                    if static == -1:
                        shape.append(runtime_shape[pos])
                    else:
                        shape.append(static)
                # The shape has to be of type int32.
                # This is very import otherwise you will get runtime
                # exceptions.
                shape = tf.to_int32(
                    tf.reshape(tf.concat(shape, axis=0), (len(shape),)))
                tensor = tf.sparse_tensor_to_dense(tensor)
                tensor = tf.reshape(tensor, shape)
            return tf.cast(tensor, dtype_run)

    def _is_static_shape(self, key):
        return -1 not in self.tensors_shape[key]

    @classmethod
    def load(cls, dir_path_records, prefix):
        file_path = os.path.join(dir_path_records, prefix + 'meta.yaml')
        if not os.path.exists(file_path):
            raise Exception('RecordMeta file {} does not exist.'.format(file_path))
        with open(file_path, 'r') as fi:
            data = yaml.load(fi, Loader=yaml.FullLoader)
        meta = cls(prefix)
        for key, value in data.items():
            setattr(meta, key, value)
        return meta


class NPGenerator(object):
    def __init__(self, np_data):
        """
        Helper class for accessing dictionarys of numpy data in the write_tfr
        method

        Parameters
        ----------
        np_data : dict
            Dictionary of numpy arrays that should be converted to tf.Examples

        Raises
        ------
        Exception
            If the number of examples is not the same for every dictionary
            entry.

        Returns
        -------
        None.

        """
        self._np_data = np_data
        default_key = list(np_data.keys())[0]
        self._num_examples = list(np_data.values())[0].shape[0]
        self.meta = {}

        for key, value in np_data.items():
            if len(value.shape) == 0:
                self.meta[key] = value
            elif value.shape[0] != self._num_examples:
                raise Exception(
                    'We have a different amount of examples '
                    '{}:{} vs {}:{}.'.format(default_key, self._num_examples,
                                              key, value.shape[0]))

        self._index = 0

    def get_meta(self):
        return self.meta

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._index < self._num_examples:
            result = {}
            for key, value in self._np_data.items():
                if key not in self.meta.keys():
                    result[key] = value[self._index]
            self._index += 1
            return result
        raise StopIteration()


def write_tfr(np_data, rw, rm, out_dir):
    """
    Create tf.Records with tf.Example out of the numpy data.

    Parameters
    ----------
    np_data :
        Either a dictionary of numpy arrays for which the first dimension is
        the example index. Alternatively dict generator or iterable object,
        which always has the same keys and each element represents one example
        and raises an StopIteration exception when done.
    rw : RecordioWriter
        A RecordioWriter object that handles writing the tf.records to file
    rm : RecordMeta
        A RecordMeta Object that stores meta information about the saved data
    out_dir : str
        The output directory.

    Raises
    ------
    Exception
        If np_data is not in one of the valid input formats.

    Returns
    -------
    None.

    """
    if isinstance(np_data, dict):
        np_data = NPGenerator(np_data)
    elif isinstance(np_data, types.GeneratorType):
        pass
    elif isinstance(np_data, collections.Iterable):
        pass
    else:
        raise Exception('np_data must be a dict, a generator or iterable.')

    for data in np_data:
        feature = {}
        for key, value in data.items():
            value_shape = list(value.shape)
            if type(value) == np.ndarray:
                if value.dtype == np.float64:
                    value = value.astype(np.float32)
                if value.dtype == bool or value.dtype == np.int32 or \
                        value.dtype == np.int16:
                    value = value.astype(np.float32)
            if key not in rm.tensors_shape:
                rm.tensors_shape[key] = value_shape
                rm.tensors_shape_max[key] = value_shape
                rm.tensors_dtype[key] = value.dtype.name
                rm.tensors_ndim[key] = value.ndim
            else:
                rm.tensors_shape[key] = \
                    _update_shape(rm.tensors_shape[key], value_shape)
                rm.tensors_shape_max[key] = \
                    map(max, zip(rm.tensors_shape_max[key], value_shape))

            _convert_np_to_features(key, value, feature)

        example = \
            tf.train.Example(features=tf.train.Features(feature=feature))
        rw.write(example.SerializeToString())

    rm.num_examples = rw.num_examples
    rm.num_records = rw.num_records
    rm.save(out_dir)


def _get_tf_dtype(np_dtype_str):
    dtypes = {}
    dtypes['uint8'] = tf.int64
    dtypes['int32'] = tf.int64
    dtypes['int64'] = tf.int64
    dtypes['float64'] = tf.float32
    dtypes['float32'] = tf.float32
    dtypes['float16'] = tf.float32
    dtypes['bool'] = tf.int64
    if np_dtype_str in dtypes:
        return dtypes[np_dtype_str]
    if np_dtype_str.startswith('str') or np_dtype_str.startswith('bytes'):
        return tf.string
    raise ValueError('Unknown data type: ' + np_dtype_str)


def _convert_np_to_features(key, value, feature):
    data_key = key + '/data'
    if value.dtype in [np.bool, bool, np.int32, np.int64]:
        val = value.ravel().tolist()
        feature[data_key] = \
            tf.train.Feature(int64_list=tf.train.Int64List(value=val))
    elif value.dtype in [np.float32, np.float64]:
        val = value.astype(np.float32)
        val = val.ravel().tolist()
        feature[data_key] = \
            tf.train.Feature(float_list=tf.train.FloatList(value=val))
    elif value.dtype.type in [np.string_, np.str_, np.bytes_]:
        val = value.ravel().tolist()
        feature[data_key] = \
            tf.train.Feature(bytes_list=tf.train.BytesList(value=val))
    else:
        raise Exception('The type ' + str(value.dtype) + ' is not supported.')

    feature = _add_dims(key, value, feature)
    return feature


def _add_dims(key, value, feature):
    shape = value.shape
    if not shape:
        # Special case for strings.
        shape = [1]
    feature[key + '/dims'] = tf.train.Feature(int64_list=tf.train.Int64List(value=shape))
    return feature


def _update_shape(tensor_shape, value_shape):
    for pos, (t_shape, v_shape) in enumerate(
            zip(tensor_shape, value_shape)):
        if t_shape != v_shape:
            # We have a changing feature shape.
            tensor_shape[pos] = -1
    return tensor_shape





