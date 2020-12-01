import _pickle as cPickle
import os
import re
import tarfile
import tensorflow as tf

CIFAR_FILENAME = 'cifar-10-python.tar.gz'
CIFAR_DOWNLOAD_URL = 'http://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME
CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'

def _download_and_extract(data_dir):
    file_path = tf.keras.utils.get_file(CIFAR_FILENAME, CIFAR_DOWNLOAD_URL, cache_subdir=data_dir, cache_dir='./datasets')
    print(file_path)
    tarfile.open(file_path, 'r:gz').extractall(os.path.join('./datasets', data_dir))
    
def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in range(1, 5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return file_names

def _read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        data_dict = cPickle.load(f, encoding='iso-8859-1')
    return data_dict

def _convert_to_tfrecord(input_files, output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = _read_pickle_from_file(input_file)
            data = data_dict['data']
            labels =  data_dict['labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                  feature={
                    'image': _bytes_feature(data[i].tobytes()),
                    'label': _int64_feature(labels[i])
                  }))
                record_writer.write(example.SerializeToString())

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecords_files(data_dir='cifar-10'):
    _download_and_extract(data_dir)
    file_names = _get_file_names()
    input_dir = os.path.join('./datasets', data_dir, CIFAR_LOCAL_FOLDER)

    for mode, files in file_names.items():
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join('./datasets', data_dir, mode+'.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write to TFRecords.
        _convert_to_tfrecord(input_files, output_file)

if __name__ == '__main__':
    create_tfrecords_files()