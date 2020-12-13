from __future__ import absolute_import
from __future__ import division
from __future__ import print_function, unicode_literals

"""
Tools to write data into tf.TFRecords and read data from tf.TFRecords.
"""

import types
import collections
import os
import shutil


try:
    import cPickle as pkl
except:
    import pickle as pkl

import json
import sys
import yaml

import numpy as np
import tensorflow as tf


DATA = 'data'
DIMS = 'dims'
NUM_EXAMPLES = 'num_examples'
NUM_RECORDS = 'num_records'
TENSORS_DTYPE = 'tensors_dtype'
TENSORS_SHAPE = 'tensors_shape'
TENSORS_SHAPE_MAX = 'tensors_shape_max'
RECORD_META_NAME = 'meta.yaml'


def load(file_path):
    _, ext = os.path.splitext(file_path)
    load_ext = {}
    load_ext['.yml'] = yaml_load
    load_ext['.yaml'] = yaml_load
    load_ext['.pkl'] = pkl_load
    load_ext['.pickle'] = pkl_load
    load_ext['.json'] = json_load
    return load_ext[ext](file_path)


def save(file_path, data, check_overwrite=False, check_create=False):
    _, ext = os.path.splitext(file_path)
    save_ext = {}
    save_ext['.yml'] = yaml_save
    save_ext['.yaml'] = yaml_save
    save_ext['.pkl'] = pkl_save
    save_ext['.pickle'] = pkl_save
    save_ext['.json'] = json_save
    return save_ext[ext](file_path, data, check_overwrite, check_create)


def yaml_load(file_path):
    return _load(file_path, lambda x: yaml.load(x, Loader=yaml.FullLoader))


def yaml_save(file_path, data, check_overwrite=False, check_create=False):
    def dump(fw, da):
        yaml.dump(da, fw)
    _save(file_path, data, dump, check_overwrite, check_create)


def pkl_load(file_path):
    return _load(file_path, pkl.load)


def pkl_save(file_path, data, check_overwrite=False, check_create=False):
    def dump(fw, da):
        pkl.dump(da, fw, pkl.HIGHEST_PROTOCOL)
    _save(file_path, data, dump, check_overwrite, check_create)


def json_load(file_path):
    return _load(file_path, json.load)


def json_save(file_path, data, check_overwrite=False, check_create=False):
    def dump(fw, da):
        json.dump(da, fw, pkl.HIGHEST_PROTOCOL)
    _save(file_path, data, dump, check_overwrite, check_create)


def _load(file_path, loader_fn):
    if not os.path.exists(file_path):
        raise Exception('No file exists at {}.'.format(file_path))
    with open(file_path, 'r') as fi:
        return loader_fn(fi)


def _save(file_path, data, saver_fn):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'w') as fo:
        saver_fn(fo, data)


def tf_record_map(dataset, dir_path_records, prefix='',
                  record_keys=None, num_threads=None,
                  output_buffer_size=None):

    record_meta = RecordMeta.load(dir_path_records, prefix)

    # We filter optionally the keys we will return.
    keys = record_meta.keys()
    if record_keys is not None:
        keys = list(set(keys) - set(record_keys))

    keys = sorted(keys)

    def _parse_function(example_proto):
        features = {}
        for key in keys:
            record_meta.add_tf_feature(key, features)

        parsed_features = tf.io.parse_single_example(example_proto, features)

        values = []
        for key in keys:
            values.append(record_meta.reshape_and_cast(key, parsed_features))
        return values

    dataset = dataset.map(_parse_function)

    return dataset, keys


def get_record_file_name(record, num_records=None):
    if num_records is not None:
        return '{0:05d}_{1:05d}.tfrecord'.format(record, num_records)
    else:
        return '{0:05d}.tfrecord'.format(record)


def get_record_meta_path(dir_path_records, prefix):
    return os.path.join(dir_path_records, prefix + RECORD_META_NAME)


def write_tfr(np_data, rw, rm, out_dir):
    """
    Create tf.Records with tf.Example out of the numpy data.

    We store the data in *.tfrecord files where the first number is the
    record index and the second one the total number of records.
    In addition we store a meta.yaml file which contains the meta information
    of the data.

    Args:

          np_data: Either a dict with numpy arrays for which the first
              dimension is the example index and it has be identical
              for all arrays. Alternatively dict generator or iterable
              object, which always has the same keys and each element
              represents one example and raises an StopIteration
              exception when done.
          rw: A RecordWriter object
          rm: a RecordMeta object
          out_dir: The directory in which we store the records.
    """
    if isinstance(np_data, dict):
        np_data = _np_generator(np_data)
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

            convert_np_to_features(key, value, feature)

        example = \
            tf.train.Example(features=tf.train.Features(feature=feature))
        rw.write(example.SerializeToString())

    rm.num_examples = rw.num_examples
    rm.num_records = rw.num_records
    rm.save(out_dir)


class RecordMeta(object):
    def __init__(self, prefix=''):
        self.num_examples = 0
        self.num_records = 0
        self.prefix = prefix
        self.tensors_dtype = {}
        self.tensors_shape = {}
        self.tensors_shape_max = {}
        self.tensors_ndim = {}

    def save(self, dir_path_records):
        # check
        file_path = get_record_meta_path(dir_path_records, self.prefix)
        data = {}
        variables = vars(self)
        for key, value in variables.items():
            if type(value) == dict:
                for k, v in value.items():
                    if type(v) == map:
                        value[k] = list(v)
            if key[0] != '_':
                data[key] = value
        save(file_path, data)

    def keys(self):
        return self.tensors_shape.keys()

    def add_tf_feature(self, key, features):
        tensor_shape = self.tensors_shape[key]
        dtype_load = _get_tf_dtype_load(self.tensors_dtype[key])
        if self._is_static_shape(key):
            features[get_data_key(key)] = tf.io.FixedLenFeature(
                dtype=dtype_load, shape=tensor_shape)
        else:
            features[get_data_key(key)] = tf.io.VarLenFeature(dtype=dtype_load)
            features[get_dim_key(key)] = tf.io.FixedLenFeature(
                shape=(self.tensors_ndim[key],),
                dtype=tf.int64)

    def _is_static_shape(self, key):
        return -1 not in self.tensors_shape[key]

    def reshape_and_cast(self, key, tensors):
        with tf.name_scope('reshape_and_cast/{}'.format(key)):
            dtype_run = _get_tf_dtype_run(self.tensors_dtype[key])

            tensor = tensors[get_data_key(key)]

            if not self._is_static_shape(key):
                runtime_shape = tensors[get_dim_key(key)]
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

    @classmethod
    def load(cls, dir_path_records, prefix):
        data = record_load_meta(dir_path_records, prefix)
        meta = cls(prefix)
        for key, value in data.items():
            setattr(meta, key, value)
        return meta


def records_to_data(dir_path_records, prefix='', keys=None):
    """Read tfrecords into python without tensorflow.

    We assume the data has been generated by np_to_records.

    Args:
         dir_path_records: The directory in which we store the records.
         prefix: name prefix for the records

    Returns:
         result: A dict with numpy arrays, the first dimension is identical
             for all arrays.
    """
    for record in record_to_np(dir_path_records, prefix, keys):
        yield record


def get_record_file_paths(dir_path_records, prefix):
    meta = RecordMeta.load(dir_path_records, prefix)
    file_paths = []
    for record in range(meta.num_records):
        fp1 = os.path.join(dir_path_records,
                           prefix + get_record_file_name(record,
                                                         meta.num_records))
        if os.path.exists(fp1):
            file_paths.append(fp1)
        else:
            fp2 = os.path.join(dir_path_records,
                               prefix + get_record_file_name(record))
            file_paths.append(fp2)

    return file_paths


def record_load_meta(dir_path_records, prefix):
    file_path = os.path.join(dir_path_records, prefix + RECORD_META_NAME)
    if not os.path.exists(file_path):
        raise Exception('RecordMeta file {} does not exist.'.format(file_path))

    return load(file_path)


def single_record_to_np(dir_path_records, prefix, max_num=-1, keys=None):
    """Read tfrecords into python without tensorflow.

    We assume the data has been generated by np_to_records.

    Args:
         dir_path_records: The directory in which we store the records.

    Returns:
         result: A dict with numpy arrays, the first dimension is identical
             for all arrays.
    """
    feature_names = None
    file_path = get_record_file_paths(dir_path_records, prefix)[0]
    out = {}
    num = 0
    for example_str in tf.python_io.tf_record_iterator(file_path):
        example = tf.train.Example()
        example.ParseFromString(example_str)

        if feature_names is None:
            feature_names = get_feature_names(example)
            if keys is not None:
                feature_names.intersection(set(keys))
            for fn in feature_names:
                out[fn] = []

        for feature_name in feature_names:
            out[feature_name] += [convert_features_to_np(feature_name,
                                                         example)]
        num += 1

        if max_num > 0 and num == max_num:
            break
    return out


def record_to_np(dir_path_records, prefix, keys=None):
    """Read tfrecords into python without tensorflow.

    We assume the data has been generated by np_to_records.

    Args:
         dir_path_records: The directory in which we store the records.

    Returns:
         result: A dict with numpy arrays, the first dimension is identical
             for all arrays.
    """
    feature_names = None
    for file_path in get_record_file_paths(dir_path_records, prefix):
        for example_str in tf.python_io.tf_record_iterator(file_path):
            example = tf.train.Example()
            example.ParseFromString(example_str)

            if feature_names is None:
                feature_names = get_feature_names(example)
                if keys is not None:
                    feature_names.intersection(set(keys))

            yield {feature_name: convert_features_to_np(feature_name, example)
                   for feature_name in feature_names}


def get_data_key(key):
    return key + '/' + DATA


def get_dim_key(key):
    return key + '/' + DIMS


def add_dims(key, value, feature):
    shape = value.shape
    if not shape:
        # Special case for strings.
        shape = [1]
    feature[get_dim_key(key)] = _int_feature(shape)
    return feature


def convert_np_to_features(key, value, feature):
    data_key = get_data_key(key)
    if value.dtype in [np.bool, bool, np.int32, np.int64]:
        feature[data_key] = _int_feature(value.ravel().tolist())
    elif value.dtype in [np.float32, np.float64]:
        feature[data_key] = _float_feature(
            value.astype(np.float32).ravel().tolist())
    elif value.dtype.type in [np.string_, np.str_, np.byte_]:
        feature[data_key] = _bytes_feature(value.tostring())
    else:
        raise Exception('The type ' + str(value.dtype) + ' is not supported.')

    feature = add_dims(key, value, feature)
    return feature


def get_feature_names(example):
    feature_names = set([])
    for feature in example.features.feature.keys():
        feature_names.add(str(feature.split('/')[0]))
    return feature_names


def convert_features_to_np(key, example):
    feature = example.features.feature
    feature_data = _convert_feature_to_np(feature[get_data_key(key)])
    shape = _convert_feature_to_np(feature[get_dim_key(key)])
    return feature_data.reshape(shape)


def _get_tf_dtype_load(np_dtype_str):
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


def _get_tf_dtype_run(np_dtype_str):
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


def _convert_feature_to_np(feature, squeeze=False):
    if feature.int64_list.ByteSize():
        res = np.array(feature.int64_list.value)
    elif feature.float_list.ByteSize():
        res = np.array(feature.float_list.value)
    elif feature.bytes_list.ByteSize():
        res = np.array(feature.bytes_list.value)
    else:
        raise Exception('No data set for ' + feature + '.')
    if squeeze:
        return np.squeeze(res)
    return res


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _update_shape(tensor_shape, value_shape):
    for pos, (t_shape, v_shape) in enumerate(
            zip(tensor_shape, value_shape)):
        if t_shape != v_shape:
            # We have a changing feature shape.
            tensor_shape[pos] = -1
    return tensor_shape


class _np_generator(object):
    def __init__(self, np_data):
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


class RecordioWriter(object):
    def __init__(self, dir_path_records, num_examples_per_record, prefix=''):
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
        self.update_file_path_records()

    def update_file_path_records(self):
        for pos, file_path_record in enumerate(self._file_path_records):
            shutil.move(file_path_record,
                        os.path.join(self._dir_path_records,
                                     self._prefix +
                                     get_record_file_name(pos,
                                                          self.num_records)))

    def write(self, serialized_example_proto):
        if self._num_examples % self._num_examples_per_record == 0:
            self._update_writer()
        self._writer.write(serialized_example_proto)
        self._num_examples += 1

    def _update_writer(self):
        file_path_record = \
            os.path.join(self._dir_path_records,
                         self._prefix +
                         get_record_file_name(self.num_records, 0))

        self._file_path_records.append(file_path_record)
        self._writer = tf.io.TFRecordWriter(file_path_record)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
