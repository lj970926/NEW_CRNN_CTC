import tensorflow as tf
import json
import os
import random
import json
import cv2


def next_batch(batch_size):
    '''

    :param batch_size: the number of pictures of per size
    :return: a tensor that consists of pictures that have a height of 32 and a label list
    '''
    #get the batch and nomoralize

    data_list = os.listdir('/home/lijin/PycharmProjects/CRNN_CTC/data')
    batch_nm = random.sample(data_list, batch_size)
    batch = []
    for name in batch_nm:
        batch.append(cv2.imread('/home/lijin/PycharmProjects/CRNN_CTC/data/' + name, 0))
    # get the label

    with open('/home/lijin/PycharmProjects/CRNN_CTC/data/label.json') as f:
        label_dict = json.load(f)
    labels = []
    for nm in batch_nm:
        labels.append(label_dict[nm])

    return batch, labels
