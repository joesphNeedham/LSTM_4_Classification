import tensorflow as tf
import numpy as np
from string import punctuation
from collections import Counter
from sklearn.model_selection import train_test_split

"Sentiment_RNN"


class DataProcess(object):
    def __init__(self, review_path, label_path):
        self.review_path = review_path
        self.label_path = label_path
        self.vocab_int_len = 0

    def integer_coding(self):
        """
        剔除掉文本中的标点字符，为reviews中的每个词进行编码。
        为label中的每个词进行整数编码。
        :return:
        """
        with open(self.review_path, "r") as f_review:
            reviews = f_review.read()
            all_text = ''.join([c for c in reviews if c not in punctuation])
            reviews = all_text.split('\n')
            all_text = ' '.join(reviews)
            words = all_text.split()
            counts = Counter(words)
            vocab = sorted(counts, key=counts.get, reverse=True)
            vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
            self.vocab_int_len = len(vocab_to_int)+1
            reviews_ints_coding = []
            for each in reviews:
                reviews_ints_coding.append([vocab_to_int[word] for word in each.split()])
        with open(self.label_path, "r") as f_labels:
            labels = f_labels.read()
            labels = labels.split("\n")
            labels_ints_coding = np.array([1 if each == 'positive' else 0 for each in labels])
        non_zero_idx = [ii for ii, review in enumerate(reviews_ints_coding) if len(review) != 0]
        reviews_ints_checked = [reviews_ints_coding[ii] for ii in non_zero_idx]
        labels_ints_checked = np.array([labels_ints_coding[ii] for ii in non_zero_idx])
        return reviews_ints_checked, labels_ints_checked

    def feature_coding(self, seq_len=200):
        """
        :param seq_len: 预设值的每句话对应的特征编码最大长度。
        :return:
        """
        reviews_ints,labels_ints = self.integer_coding()
        features = np.zeros((len(reviews_ints), seq_len), dtype=int)
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:seq_len]
        train_x, val_x = train_test_split(features,train_size=0.8,random_state=1)
        train_y, val_y = train_test_split(labels_ints,train_size=0.8,random_state=1)
        val_x,test_x = train_test_split(val_x,train_size=0.5,random_state=1)
        val_y,test_y = train_test_split(val_y,train_size=0.5,random_state=1)
        return [(train_x, train_y),(test_x,test_y),(val_x, val_y)]

# processer = DataProcess("../sentiment-network/reviews.txt","../sentiment-network/labels.txt")
# features = processer.feature_coding()


class Graph(object):
    def __init__(self, review_path,label_path):
        self.lstm_state_size = 256
        self.lstm_layer = 1
        self.batch_size = 500
        self.learning_rate = 0.001
        self.embed_size = 300
        self.keep_prob = 0.5
        self.epochs = 10
        self.data_processor = DataProcess(review_path, label_path)
        self.features = self.data_processor.feature_coding()
        self.word_size = self.data_processor.vocab_int_len
    @staticmethod
    def get_batches(x, y, batch_size=100):
        """
        :param x: 特征
        :param y: 标签
        :param batch_size: 训练集规模
        :return:
        """
        n_batches = len(x) // batch_size
        x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
        for ii in range(0, len(x), batch_size):
            yield x[ii:ii + batch_size], y[ii:ii + batch_size]

    def train(self):
        train_x, train_y = self.features[0]
        val_x,val_y = self.features[2]
        graph = tf.Graph()
        with graph.as_default():
            inputs_ = tf.placeholder(tf.int32, [None, None], name="input")
            labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            embedding = tf.Variable(tf.random_uniform((self.word_size, self.embed_size), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs_)
            lstm =tf.nn.rnn_cell.BasicLSTMCell(self.lstm_state_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([drop] * self.lstm_layer)
            initial_state = cell.zero_state(self.batch_size, tf.float32)
            outputs,final_state = tf.nn.dynamic_rnn(cell, embed,
                                             initial_state=initial_state)
            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
            cost = tf.losses.mean_squared_error(labels_, predictions)

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
            correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            saver = tf.train.Saver()
        with tf.Session(graph=graph) as sess:
            initial = tf.global_variables_initializer()
            sess.run(initial)
            iteration = 1
            for e in range(self.epochs):
                for ii,(x,y) in enumerate(self.get_batches(train_x,train_y,self.batch_size),1):
                    feed = {inputs_:x,
                            labels_: y[:,None],
                            keep_prob: self.keep_prob
                            }
                    loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, self.epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))
                    if iteration % 25 == 0:
                        val_acc = []
                        for x, y in self.get_batches(val_x, val_y, self.batch_size):
                            feed = {inputs_: x,
                                    labels_: y[:, None],
                                    keep_prob: 1}
                            batch_acc= sess.run([accuracy], feed_dict=feed)
                            val_acc.append(batch_acc)
                            print("Val acc: {:.3f}".format(np.mean(val_acc)))
                    iteration +=1
            saver.save(sess, "checkpoints/sentiment_rnn.ckpt")

    def test(self):
        graph = tf.Graph()
        with graph.as_default():
            inputs_ = tf.placeholder(tf.int32, [None, None], name="input")
            labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            embedding = tf.Variable(tf.random_uniform((self.word_size, self.embed_size), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs_)
            lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_state_size)
            drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=self.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([drop] * self.lstm_layer)
            initial_state = cell.zero_state(self.batch_size, tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                     initial_state=initial_state)
            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
            cost = tf.losses.mean_squared_error(labels_, predictions)

            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
            correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            test_x, test_y = self.features[1]
            test_acc = []
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            # saver = tf.train.import_meta_graph('checkpoints/sentiment_rnn.ckpt.meta')
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            for ii, (x, y) in enumerate(self.get_batches(test_x, test_y, self.batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 1
                        }
                batch_acc = sess.run([accuracy], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


if __name__ == "__main__":
    lstm = Graph("../sentiment-network/reviews.txt", "../sentiment-network/labels.txt")
    # lstm.train()
    lstm.test()
