import model_avl_tree
from utils import load_data, get_code_dict, word2num, build_tree, load_data_2_root,merge_corpus,get_idf
import utils
import time

if __name__ == "__main__":
    root, stopwords, filename = None, utils.get_stopwords(), 'data/kejiao.txt'
    data = load_data(filename, stopwords)
    word_number, word_freq, number_word = get_code_dict(data)
    data_code, op = word2num(data, word_number), model_avl_tree.OperationTree()
    root = build_tree(op, word_freq, root)
    vocab, num_docs = merge_corpus(data_code)
    idf_list = get_idf(num_docs, vocab)

    time_start = time.time()
    load_data_2_root(op, root, data_code)
    time_end = time.time()
    print(time_end - time_start)

    result, add_word = op.find_word_tfidf(root, number_word, idf_list,20)
    print(add_word)
