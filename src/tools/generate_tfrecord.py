import tensorflow as tf
import json
import cv2
import os
import random
import sys

pro_dir = '/home/lijin/PycharmProject/CRNN_CTC/'
label_dir = pro_dir + '/json/label.json'
map_dir = pro_dir + '/json/char_map.json'
data_dir = pro_dir + 'data'

img_height = 32
validation_split_fraction = 0.1


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(tf.train.Int64List(value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(tf.train.BytesList(value))


def _label_to_string(label):
    with open(map_dir, 'r') as f:
        map_dict = json.load(f)
        int_list = []
        for c in label:
            map_dict[c] = len(map_dict)
            int_list.append(map_dict[c])

    return int_list


def _write_tfrecord(record_name, file_name_list):
    with tf.python_io.TFRecordWriter(pro_dir) as writer:
        f = open(label_dir, 'r')
        label_dict = json.load(f)
        for name in file_name_list:
            image_path = data_dir + '/' + name
            image = cv2.imread(image_path)
            label = label_dict[name]
            if image is None:
                continue
            h, w, c = image.shape
            width = int(w * img_height / h)
            cv2.resize(image, (width, img_height))
            is_success, image_buffer = cv2.imencode('.jpg', image)
            if not is_success:
                continue
            features = tf.train.Features(feature={
                'label': _int64_feature(_label_to_string(label)),
                'images': _bytes_feature(image_buffer.tostring()),
                'imgname': _bytes_feature(name)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>>Writing to {:s}.tfrecords'.format(record_name))
            sys.stdout.flush()
            sys.stdout.write('\n')
            sys.stdout.write('>> {:s}.tfrecords write finish.'.format(record_name))
            sys.stdout.flush()
        f.close()


def genetate_tfrecord():
    file_name_list = os.listdir(data_dir)
    random.shuffle(file_name_list)
    split_index = int(len(file_name_list) * (1 - validation_split_fraction))
    data_divide = {'train': file_name_list[:split_index - 1],
                   'validation': file_name_list[split_index:]}
    for name in ['train', 'validation']:
        _write_tfrecord(name, data_divide[name])
