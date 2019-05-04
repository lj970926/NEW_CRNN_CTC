import tensorflow as tf
import numpy as np
from src.model import model
import os
import json
import time

# some paths
pro_dir = '/home/lijin/PycharmProjects/CRNN_CTC/'
label_dir = pro_dir + '/json/label.json'
map_dir = pro_dir + '/json/char_map.json'
data_dir = pro_dir + 'data'
tfrecord_path = pro_dir + 'tfrecord/validation.tfrecord'
mod_dir = pro_dir + 'ckpt/'

# some hyperprarameters
batch_size = 16
num_threads = 4
hidden_units = 256
hidden_layers = 2
learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.8
train_step = 20000
per_save = 1000
per_eval = 100


def sparse_matrix_to_list(sparse_matrix, char_map_dict=None):
    '''
    input an sparsetensor,converse it to a list that contains strings of each picture in a batch
    :param sparse_matrix: a sparseTensor instance
    :param char_map_dict: map an int value to a char value
    :return: a list that contains strings for each line
    '''
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape
    if char_map_dict is None:
        char_map_dict = json.load(open(map_dir, 'r'))
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')
    dense_matrix = len(char_map_dict.keys()) * np.ones(dense_shape, dtype=np.int32)
    for i, indice in enumerate(indices):
        dense_matrix[indice[0], indice[1]] = values[i]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(int_to_string(val, char_map_dict))
        string_list.append(''.join(s for s in string if s != '*'))
    return string_list


def int_to_string(value, char_map_dict=None):
    '''
    given a char value return the int value relevanted to it
    :param value:
    :param char_map_dict:
    :return:
    '''
    if char_map_dict is None:
        json.load(open(map_dir, 'r'))
    assert (isinstance(char_map_dict, dict) and 'char_map_dict is not a dict')

    for key in char_map_dict.keys():
        if char_map_dict[key] == int(value):
            return str(key)
        elif len(char_map_dict.keys()) == int(value):
            return ""
    raise ValueError('char map dict not has {:d} value. convert index to char failed.'.format(value))


def read_tfrecord(path, num_epochs=None):
    '''
    read images labels
    :param path:
    :param num_epochs:
    :return:
    '''
    if not os.path.exists(path):
        raise ValueError("failed to find tfrecord file in:{:s}".format(path))

    name_quene = tf.train.string_input_producer([path], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(name_quene)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.VarLenFeature(tf.int64),
        'images': tf.FixedLenFeature([], tf.string),
        'imgname': tf.FixedLenFeature([], tf.string)
    })
    images = tf.image.decode_jpeg(features['images'])
    images.set_shape([32, None, 3])
    images = tf.cast(images, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    sequence_length = tf.cast(tf.shape(images)[-2] / 4, tf.int32)
    imgname = features['imgname']
    return images, label, sequence_length, imgname


def validate_model():
    images, label, sequence_length, imgnames = read_tfrecord(path=tfrecord_path)

    batch_images, batch_labels, batch_sequence_lengths, batch_imgnames = tf.train.batch(
        tensors=[images, label, sequence_length, imgnames],
        batch_size=batch_size, dynamic_pad=True,
        capacity=1000 + 2 * batch_size,
        num_threads=num_threads)
    input_images = tf.placeholder(tf.float32, shape=[batch_size, 32, None, 3], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='input_sequence_lengths')

    char_map_dict = json.load(open(map_dir, 'r'))

    crnn_net = model.CRNNModel(phase='test',
                               hidden_num=hidden_units,
                               layer_num=hidden_layers,
                               class_num=len(char_map_dict.keys()) + 1)
    with tf.variable_scope('CRNN_CTC', reuse=False):
        model_output = crnn_net.build_network(images=input_images, sequence_length=input_sequence_lengths)

    ctc_decoded, ctc_log_prob = tf.nn.ctc_beam_search_decoder(model_output, input_sequence_lengths,
                                                              merge_repeated=False)

    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint(mod_dir)
    test_count = 0
    for record in tf.python_io.tf_record_iterator(tfrecord_path):
        test_count += 1
    step_num = test_count // batch_size

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        saver.restore(sess=sess, save_path=save_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        accuracy = []

        for _ in range(step_num):
            imgs, lbls, seq_lens, names = sess.run(
                [batch_images, batch_labels, batch_sequence_lengths, batch_imgnames])
            preds = sess.run(ctc_decoded,
                             feed_dict={input_images: imgs, input_labels: lbls, input_sequence_lengths: seq_lens})

            preds = sparse_matrix_to_list(preds[0])
            lbls = sparse_matrix_to_list(lbls)

            # print(preds)
            # print(lbls)
            for index, lbl in enumerate(lbls):
                pred = preds[index]
                total_count = len(lbl)
                correct_count = 0
                try:
                    for i, tmp in enumerate(lbl):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            for index, img in enumerate(imgs):
                print(
                    'Predict {:s} image with gt label: {:s} <--> predict label: {:s}'.format(names[index].decode('utf-8'), lbls[index],
                                                                                             preds[index]))

        accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        print('Mean test accuracy is {:5f}'.format(accuracy))

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)


if __name__ == "__main__":
    validate_model()
