import cv2
from matplotlib import pyplot as plt

pro_dir = '/home/lijin/PycharmProjects/CRNN_CTC/'
data_dir = pro_dir + 'data'
image = cv2.imread(data_dir + '/294.png')
img_height = 32
h, w, c = image.shape
print(h)
print(w)
print(c)
cv2.namedWindow('origin picture', cv2.WINDOW_AUTOSIZE)
cv2.imshow('origin picture', image)
cv2.waitKey()
width = int(w * img_height / h)
image = cv2.resize(image, (width, img_height))
h, w, c = image.shape
print(h)
print(w)
print(c)
cv2.namedWindow('resized picture', cv2.WINDOW_AUTOSIZE)
cv2.imshow('resized picture', image)
cv2.waitKey()
