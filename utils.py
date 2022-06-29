import math
import random
from collections import Counter
import jieba
import model_avl_tree


def get_stopwords():
    '''
    Get a list of stop words
    获取停用词列表
    :return:
    '''
    with open('data/stopword.txt', 'r', encoding='utf-8') as f:
        stopword = [line.strip() for line in f]
    return set(stopword)


def generate_ngram(input_list, n):
    '''
    generate ngrams
    生成 ngrams
    :param input_list:
    :param n:
    :return:
    '''
    result = []
    for i in range(1, n + 1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    return result


def load_data(filename, stopwords):
    """
    Two-dimensional array, [[sentence 1 participle list], [sentence 2 participle list],...,[sentence n participle list]]
    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            word_list = [x for x in jieba.cut(line.strip(), cut_all=False) if x not in stopwords]
            data.append(word_list)
    return data


def get_code_dict(data):
    """
    Build Chinese words into code
    将中文单词构建成code
    """
    word_number = {}
    number_word = {}
    word_freq = {}
    word_list = []
    for item in data:
        if len(item) > 0:
            [word_list.append(word) for word in item]
    word_dict = Counter(word_list)
    i = 0
    for key in word_dict:
        i = i + 1
        word_number[key] = i
        word_freq[i] = word_dict[key]
        number_word[i] = key
    return word_number, word_freq, number_word


def word2num(data, word_number):
    """
    Convert Chinese words into serial numbers
    将中文单词转化成序号
    """
    result_list = []
    for item in data:
        word_num_list = []
        if len(item) > 0:
            for word in item:
                word_num_list.append(word_number[word])
            result_list.append(word_num_list)
    return result_list


def load_data_2_root(op, root, data):
    '''
    insert node
    插入节点
    :param op:
    :param root:
    :param data:
    :return:
    '''
    for word_list in data:
        ngrams = generate_ngram(word_list, 3)
        for d in ngrams:
            op.add(root, d)
            if len(d) == 3:
                op.add_suffix(root, d)


def build_tree(op, word_freq, root):
    '''
    Build a balanced binary tree of order 1
    构建1阶平衡二叉树
    :param op:
    :param word_freq:
    :param root:
    :return:
    '''
    val_list = list(word_freq.keys())
    random.shuffle(val_list)
    for val in val_list:
        root = op.insert_bal_foo(root, val, rank=1)
        flag, parent_node = op.query_foo(root, val)
        if flag:
            node = model_avl_tree.TreeNode(0, rank=2)
            node.parent = parent_node
            parent_node.child = node
    return root


def merge_corpus(corpus):
    """
    Count the corpus, output the vocabulary, and count the number of documents that contain each word.
    统计语料库，输出词表，并统计包含每个词的文档数。
    """
    vocab = {}
    for sentence in corpus:
        words = set(sentence)
        for word in words:
            vocab[word] = vocab.get(word, 0.0) + 1.0
    return vocab, len(corpus)


def get_idf(num_docs, vocab):
    """
    Calculate the IDF value
    计算 IDF 值
    """
    idf_dict = {}
    for term in vocab:
        idf_dict[term] = math.log(num_docs / (vocab[term] + 1.0))
    return idf_dict
