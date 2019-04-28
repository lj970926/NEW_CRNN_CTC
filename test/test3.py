import tensorflow as tf
import json
import random
import os
from matplotlib import pyplot as plt
import cv2
img = cv2.imread('/home/lijin/PycharmProjects/CRNN_CTC/data/9242.jpeg')
wind = cv2.namedWindow("original picture",cv2.WINDOW_NORMAL)
cv2.imshow('original picture',img)
cv2.waitKey()
normal_img = tf.image.resize_images(img,[32,-1],0)
plt.imshow(normal_img)
