import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np

x = np.random.rand(2, 4, 5)
fw_cell_list = [rnn.BasicLSTMCell(3, forget_bias=1.0) for i in range(3)]
bw_cell_list = [rnn.BasicLSTMCell(3, forget_bias=1.0) for i in range(3)]
ret1, ret2, ret3 = rnn.stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list, x, sequence_length=[4, 4],
                                                       dtype=tf.float64)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r1, r2, r3 = sess.run([ret1, ret2, ret3])
    print(r1[0])
    print(r1[1])
    print(r2)
    print(r3)
