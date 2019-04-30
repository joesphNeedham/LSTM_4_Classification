import tensorflow as tf
import numpy as np

class lstm(object):
    def __init__(self,time_step,word_embedding_dim,lstm_state_output_dim):
        self.batch = tf.placeholder(dtype=tf.float32, shape=[None, time_step, word_embedding_dim])
        self.seq_length = tf.placeholder([None])
        self.basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_state_output_dim)
    def session(self):
        tf.nn.dynamic_rnn(self.basic_cell,self.batch,self.seq_length,dtype=tf.float32)

