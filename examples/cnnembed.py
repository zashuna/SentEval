import numpy as np
import tensorflow as tf
import logging
import os
import nltk
import cPickle
import pdb
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_CNNEMBED = '/home/shunan/Code/CNNEmbed'
VOCAB_SIZE = 483019
ZERO_IND = 483018

# import models
sys.path.insert(0, PATH_TO_CNNEMBED)
from models.CNNEmbed import CNNEmbed
sys.path.insert(0, PATH_SENTEVAL)
import senteval
tknzr = nltk.tokenize.TweetTokenizer()

def load_model():
    '''
    Load the CNN model
    '''

    # Model parameters here.
    context_len = 5
    batch_size = 50
    num_filters = 1000
    filter_size = 5
    num_layers = 11
    pos_words_num = 5
    neg_words_num = 30
    num_residual = 1
    k_max = 3
    max_doc_len = 50
    embed_dim = 300
    batch_norm = False

    cnn_model = dict()

    doc2vec_graph = tf.Graph()
    with doc2vec_graph.as_default(), tf.device("/cpu:0"):
        indices_data_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None])
        indices_target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, pos_words_num + neg_words_num])

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, embed_dim], dtype=tf.float32, trainable=True)
        inputs = tf.gather(embedding, indices_data_placeholder)
        inputs = tf.expand_dims(inputs, 3)
        inputs = tf.transpose(inputs, [0, 2, 1, 3])

        targets_embeds = tf.gather(embedding, indices_target_placeholder)
        targets_embeds = tf.expand_dims(targets_embeds, 3)
        targets_embeds = tf.transpose(targets_embeds, [0, 2, 1, 3])

        target_place_holder = tf.placeholder(tf.float32, [None, pos_words_num + neg_words_num])
        # Placeholder for training
        keep_prob_placeholder = tf.placeholder(dtype=tf.float32, name='dropout_rate')
        is_training_placeholder = tf.placeholder(dtype=tf.bool, name='training_boolean')

        # build model
        _docCNN = CNNEmbed(
            inputs, targets_embeds, target_place_holder, is_training_placeholder, keep_prob=keep_prob_placeholder,
            max_doc_len=max_doc_len, embed_dim=embed_dim, num_layers=num_layers, num_filters=num_filters,
            residual_skip=num_residual, k_max=k_max, filter_size=filter_size, batch_norm=batch_norm)

        # input of the test (supervised learning) process
        model_output = tf.squeeze(_docCNN.res)

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_docCNN = tf.Session(config=session_conf)
        saver = tf.train.Saver()

    # Restore the weights
    saver.restore(sess_docCNN, os.path.join(PATH_TO_CNNEMBED, 'latest_model_gbw', 'combined_model-1'))

    cnn_model['sess'] = sess_docCNN
    cnn_model['model'] = _docCNN
    cnn_model['placeholders'] = [indices_data_placeholder, is_training_placeholder, keep_prob_placeholder]
    cnn_model['model_output'] = model_output

    return cnn_model

def prepare(params, samples):

    return

def convert_to_tokens(cnn_model, sentence, doc_len=None):

    sentence = ' '.join(sentence)
    sentence = nltk.word_tokenize(' '.join(tknzr.tokenize(sentence)))
    sentence = [word.lower() for word in sentence]
    tokens = []

    for word in sentence:
        if word in cnn_model['word_to_index']:
            tokens.append(cnn_model['word_to_index'][word])
        else:
            tokens.append(cnn_model['word_to_index']['<unk>'])

    if len(tokens) < 5:
        # pad with zeros
        tokens = [ZERO_IND for _ in range(5 - len(tokens))] + tokens

    if doc_len is not None:
        # pad or trucate to that length
        if len(tokens) > doc_len:
            tokens = tokens[:doc_len]
        elif len(tokens) < doc_len:
            tokens = [ZERO_IND for _ in range(doc_len - len(tokens))] + tokens

    return np.reshape(np.array(tokens), (1, len(tokens)))

def batcher(params, batch):

    cnn_model = params['CNNEmbed']
    mat = []
    if 'doc_len' in params:
        all_tokens = []
        for sentence in batch:
            tokenized_sen = convert_to_tokens(cnn_model, sentence, doc_len=params['doc_len'])
            all_tokens.append(tokenized_sen)

        all_tokens = np.squeeze(np.array(all_tokens), axis=1)
        feed_dict = {cnn_model['placeholders'][0]: all_tokens, cnn_model['placeholders'][2]: 1.,
                     cnn_model['placeholders'][1]: False}
        encoding = cnn_model['sess'].run(cnn_model['model_output'], feed_dict)
        mat = encoding
    else:
        for sentence in batch:
            tokenized_sen = convert_to_tokens(cnn_model, sentence)

            # Feed it through model
            feed_dict = {cnn_model['placeholders'][0]: tokenized_sen, cnn_model['placeholders'][2]: 1.,
                         cnn_model['placeholders'][1]: False}
            encoding = cnn_model['sess'].run(cnn_model['model_output'], feed_dict)
            mat.append(encoding)

    return np.array(mat)


# define senteval params
params_cnn = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
params_cnn['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 64, 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':

    # Load CNNEmbed model
    cnn_model = load_model()
    use_padding = False

    with open(os.path.join(PATH_TO_CNNEMBED, 'gbw_cache/word_to_index.pkl'), 'r') as f:
        word_to_index = cPickle.load(f)

    cnn_model['word_to_index'] = word_to_index
    params_cnn['CNNEmbed'] = cnn_model

    if use_padding:
        doc_lengths = [15, 24, 32, 41, 47]
        for doc_len in doc_lengths:
            print('Using document length {}'.format(doc_len))
            params_cnn['doc_len'] = doc_len

            se = senteval.engine.SE(params_cnn, batcher, prepare)
            transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']
            results = se.eval(transfer_tasks)
            print(results)
    else:
        se = senteval.engine.SE(params_cnn, batcher, prepare)
        transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']
        results = se.eval(transfer_tasks)
        print(results)
