import numpy as np
import tensorflow as tf
import logging
import os
import pickle
import pdb
import sys

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_VAE = '/home/shunan/Code/VAEMultiSentEmbeds'

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# import models
sys.path.insert(0, PATH_TO_VAE)
from model import Model
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def load_dict(dict_file, vocab_size):
    """
    Load the mapping from tokens to ID

    :param dict_files: (str) The file containing the table of word frequencies.
    :param vocab_sizes: (int) The size of the vocabulary. If None, load the entire vocabulary
    :return (dict) mapping from word to ID/index in word embedding matrix.
    """

    mapping = dict()
    vocab_size = vocab_size if vocab_size is not None else float('Inf')
    with open(dict_file, 'r') as f:
        ind = 0
        line = f.readline()
        while line != '' and ind < vocab_size:
            line = line.strip().split('\t')
            mapping[line[0]] = ind

            line = f.readline()
            ind += 1

    return mapping

def prepare(params, samples):

    return


def convert_to_inds(sents, word_to_index):
    """
    Convert a list of strings to indices in the embedding matrix.

    :param sents: (list) list of sentences, as strings.
    :param lang: (str) The language
    :return: Tuple of numpy array, first being the array of indices and the second the sentence lengths.
    """

    doc_inds = []
    lengths = []
    max_len = 0
    for sent in sents:
        # already pre-processed, so not much to do here.
        sent.append('<eos>')

        sent_inds = []
        unk_ind = word_to_index['<unk>']
        for token in sent:
            ind = word_to_index.get(token.lower(), unk_ind)
            sent_inds.append(ind)

        lengths.append(len(sent_inds))
        if len(sent_inds) > max_len:
            max_len = len(sent_inds)
        doc_inds.append(sent_inds)

    # pad to max length
    for i in range(len(doc_inds)):
        doc_inds[i] = doc_inds[i] + [0 for _ in range(max_len - len(doc_inds[i]))]

    return np.array(doc_inds), np.array(lengths)


def batcher(params, batch):

    model = params['VAE']
    word_ids, lengths = convert_to_inds(batch, params['word2idx'])
    feed_dict = {
        model.placeholders['word_ids_l0']: word_ids,
        model.placeholders['sent_lengths_l0']: lengths,
        model.placeholders['is_training']: False
    }

    # Feed it through model
    encoding = params['sess'].run(model._sentence_embeddings[0][0], feed_dict)

    return encoding


if __name__ == '__main__':

    # load model parameters
    with open(os.path.join(PATH_TO_VAE, 'saved_models/en_params.pkl'), 'rb') as f:
        params = pickle.load(f)

    model = Model(params)
    model.build_encoder()
    model.build_decoder()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(PATH_TO_VAE, 'saved_models/multilingual_VAE-1'))

    # define senteval params
    params_sent_eval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
    params_sent_eval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 64, 'tenacity': 3, 'epoch_size': 2}
    params_sent_eval['sess'] = sess

    vocab_size = 30219
    word_to_index = load_dict('/home/shunan/Data/europarl/dict_files/de-en.en.dict', vocab_size)

    params_sent_eval['VAE'] = model
    params_sent_eval['word2idx'] = word_to_index

    se = senteval.engine.SE(params_sent_eval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']
    results = se.eval(transfer_tasks)
    print(results)
