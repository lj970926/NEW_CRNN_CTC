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
tfrecord_path = pro_dir + 'tfrecord/train.tfrecord'
mod_dir = pro_dir + 'logs/'

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
        char_map_dict=json.load(open(map_dir, 'r'))
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


def train_model():
    images, label, sequence_length, _ = read_tfrecord(path=tfrecord_path)

    batch_images, batch_labels, batch_sequence_lengths = tf.train.batch(tensors=[images, label, sequence_length],
                                                                        batch_size=batch_size, dynamic_pad=True,
                                                                        capacity=1000 + 2 * batch_size,
                                                                        num_threads=num_threads)
    input_images = tf.placeholder(tf.float32, [batch_size, 32, None, 3], name='input_images')
    input_labels = tf.sparse_placeholder(tf.int32, name='input_labels')
    input_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='input_sequence_lengths')
    char_map_dict = json.load(open(map_dir, 'r'))
    crnn_net = model.CRNNModel(phase='train',
                               hidden_num=hidden_units,
                               layer_num=hidden_layers,
                               class_num=len(char_map_dict) + 1)  # contain space
    with tf.variable_scope('CRNN_CTC', reuse=False):
        model_output = crnn_net.build_network(images=input_images, sequence_length=input_sequence_lengths)

    ctc_loss = tf.reduce_mean(
        tf.nn.ctc_loss(labels=input_labels, inputs=model_output, sequence_length=input_sequence_lengths,
                       ignore_longer_outputs_than_inputs=True))

    ctc_decoded, ctc_log_prob = tf.nn.ctc_beam_search_decoder(model_output, input_sequence_lengths,
                                                              merge_repeated=False)
    sequence_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0], tf.int32), input_labels))
    global_step = tf.train.create_global_step()
    learn_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate,staircase=True)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learn_rate).minimize(loss=ctc_loss, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # set tf summary

    tf.summary.scalar(name='CTC_Loss', tensor=ctc_loss)
    tf.summary.scalar(name='Learning_Rate', tensor=learn_rate)
    tf.summary.scalar(name='Seqence_Distance', tensor=sequence_distance)
    merge_summary_op = tf.summary.merge_all()

    # set checkout saver
    saver = tf.train.Saver()
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'crnn_ctc_ocr_{:s}.ckpt'.format(str(train_start_time))
    model_path = pro_dir + 'ckpt/' + model_name
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:

        summary_writer = tf.summary.FileWriter(mod_dir)
        summary_writer.add_graph(sess.graph)

        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(train_step):
            imgs, lbls, seq_lens = sess.run([batch_images, batch_labels, batch_sequence_lengths])

            _, cl,lr, sd, preds, summary,modop = sess.run(
                [optimizer, ctc_loss,learn_rate, sequence_distance, ctc_decoded, merge_summary_op,model_output],
                feed_dict={input_images: imgs, input_labels: lbls, input_sequence_lengths: seq_lens})

            if (step + 1) % per_save == 0:
                summary_writer.add_summary(summary=summary, global_step=step)
                saver.save(sess=sess, save_path=model_path, global_step=step)

            if (step + 1) % per_eval == 0:
                predd = sparse_matrix_to_list(preds[0])
                gt_labels = sparse_matrix_to_list(lbls)
                accuracy = []

                for index, gt_label in enumerate(gt_labels):
                    pred = predd[index]
                    total_count = len(gt_label)
                    correct_count = 0
                    try:
                        for i, tmp in enumerate(gt_label):
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
                accuracys = np.mean(np.array(accuracy).astype(np.float32), axis=0)

                print(
                    'step:{:d} learning_rate={:9f} ctc_loss={:9f} sequence_distance={:9f} train_accuracy={:9f}'.format(
                        step + 1, lr, cl, sd, accuracys))
        summary_writer.close()

        # stop file queue
        coord.request_stop()
        coord.join(threads=threads)


if __name__ == "__main__":
    train_model()
