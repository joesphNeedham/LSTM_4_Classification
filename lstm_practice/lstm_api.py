
"""
1. 每个词通过embeding后，可以由一个一维向量表示。
2. 一句话通过embedding后，可以由一个矩阵表示。
3. X_batch中表示的是一个文本中的4句话被向量化后的结果。
"""


import tensorflow as tf
import numpy as np


class LSTM(object):
    def __init__(self, time_step, word_embed,state_dim):
        """
        :param:the dimension of output and state of lstm cell（lstm对应输出的状态和output维度）
        :param time_step:the number of words in a sentence (一个句子中有多少个词或者说是分割后的词)
        :param word_embed: the embedding dimension of a word or phrase（一个词对应的维度）
        :return:
        """
        self.batch = tf.placeholder(dtype=tf.float32, shape=[None, time_step, word_embed])
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=state_dim, state_is_tuple=True)
        self.sentence_length = tf.placeholder(tf.int32, [None])
        # self.inital_state = self.lstm_cell.zero_state(self.batch.shape[0],dtype=tf.float32)

    def session(self, input_data, seq_length):
        outputs, states = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                            inputs=self.batch,
                                            sequence_length=self.sentence_length,
                                            # initial_state=self.inital_state,
                                            dtype=tf.float32)
        with tf.Session() as sess:
            initalize = tf.global_variables_initializer()
            sess.run(initalize)
            output_val, states_val = sess.run([outputs, states], feed_dict={self.batch: input_data,
                                                                            self.sentence_length: seq_length})
            return output_val, states_val


if __name__ == "__main__":
    X = np.array([
        [[0, 1, 2], [9, 8, 7]],
        [[3, 4, 5], [0, 0, 0]],
        [[6, 7, 8], [6, 5, 4]],
        [[9, 0, 1], [3, 2, 1]]
    ])
    TIME_STEP = 2
    WORD_EMBED = 3
    STATE_DIM = 5
    sequence_length = np.array([2,1,2,2])
    lstm = LSTM(TIME_STEP, WORD_EMBED, STATE_DIM)
    out, state = lstm.session(input_data=X,seq_length=sequence_length)
    print(u"每个时间步长对应的输出")
    print(out)
    print("state:")
    print(state)
    print("每个时间步长的状态")
    print(state.h)
    print("---------")
    print(out[:, -1])

