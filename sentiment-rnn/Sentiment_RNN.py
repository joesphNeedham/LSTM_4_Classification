import tensorflow as tf
import numpy as np
from string import punctuation
from collections import Counter


def process_reviews(path):
    """
    :param path:
    :return: [review1,review2,review3]
    剔除掉文本中的标点字符，得到每个评论构成的list
    """
    with open(path, "r") as f:
        reviews = f.read()
        all_text = ''.join([c for c in reviews if c not in punctuation])
        reviews = all_text.split('\n')
        return reviews


def coding_labels(path):
    with open(path, "r") as f:
        labels = f.read()
        labels = labels.split("\n")
        labels = np.array([1 if each == 'positive' else 0 for each in labels])
        return labels


def coding_reviews_words(path):
    """
    为评论中的每个词进行编码
    :param path: reviews数据所在的路径
    :return: 关于各个评论数值编码好的列表
    """
    reviews = process_reviews(path)
    all_text = ' '.join(reviews)
    words = all_text.split()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    reviews_ints_coding = []
    for each in reviews:
        reviews_ints_coding.append([vocab_to_int[word] for word in each.split()])
    return reviews_ints_coding


def noise_data_process(reviews_ints, labels):
    """
    统计评论中哪些评论的长度为0，剔除掉。
    :param reviews_ints:
    :param labels:
    :return:
    """
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]
    reviews_ints_checked = [reviews_ints[ii] for ii in non_zero_idx]
    labels_checked = np.array([labels[ii] for ii in non_zero_idx])
    return reviews_ints_checked, labels_checked

if __name__ == "__main__":
    reviews_ints = coding_reviews_words("../sentiment-network/reviews.txt")
    print(reviews_ints[1])