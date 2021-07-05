# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import logging
import torch.nn as nn
import os
from argparse import ArgumentParser
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
np.random.seed(42)  # for reproducibility

_LOGGER_LEVEL = logging.INFO
_LOGGER_FORMAT = "%(asctime).10s - %(levelname)s - %(name)s - %(message)s"

# -*- coding: utf-8 -*-


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.lowest_loss = 100000.0
        self.early_stop = False
        self.val_loss_max = 100000.0

#    def __call__(self, val_F1, model):
#
#        score = val_F1
#
#        if self.best_score is None:
#            self.best_score = score
#            self.save_checkpoint(val_F1, model)
#
#        elif score <= self.best_score:
#            self.counter += 1
#            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#            if self.counter >= self.patience:
#                self.early_stop = True
#        else:
#            self.best_score = score
#            print(f'F1  ({self.val_F1_max:.6f} --> {val_F1:.6f}).  Saving model ...')
#            self.save_checkpoint(val_F1, model)
#            self.counter = 0

    def __call__(self, val_loss, model):
        loss = val_loss
#        import pdb;pdb.set_trace()
#        if self.lowest_loss is None:
#            self.lowest_loss = loss
#            self.save_checkpoint(val_loss, model)

        if loss >= self.lowest_loss:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Loss  ({self.lowest_loss:.6f} --> {val_loss:.6f}). ')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(
                f'Loss  ({self.lowest_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.lowest_loss = loss
#            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Loss decreased ({self.lowest_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_F1_max = val_loss


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end='')


def get_logger(command_name):
    logging.basicConfig(
        level=_LOGGER_LEVEL, stream=TqdmStream, format=_LOGGER_FORMAT)
    return logging.getLogger(command_name)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def f2_score(y_true, y_pred):

    y_true = tf.cast(y_true, "int32")
    # implicit 0.5 threshold via tf.round
    y_pred = tf.cast(tf.round(y_pred), "int32")

    y_correct = y_true * y_pred

    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)

    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true

    f_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    try:
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    except BaseException:
        f_score = tf.where(
            tf.math.is_nan(f_score),
            tf.zeros_like(f_score),
            f_score)

    return tf.reduce_mean(f_score)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) +
                          input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def get_args():
    parser = ArgumentParser(description='PyTorch Event Detection')
    parser.add_argument(
        '--embeddings',
        default='google',
        help='embeddings type: google, deps, gigaword, numberbatch, bert')
    parser.add_argument('--run', default=1,
                        metavar='RUN', help='run number/details')
    parser.add_argument('--gold_test', default='results/ACE_gold2.tbf',
                        help='test gold file')
    parser.add_argument('--model',
                        default='CNN2019',
                        nargs='?',
                        choices=['CNN2015', 'CNN2018', 'CNN2019', 'attention'],
                        help='list models (default: %(default)s)')
    parser.add_argument('--write_results', default=0,
                        help='0 - no, 1 - yes')
    parser.add_argument('--multi-gpu', default=0,
                        help='If GPU is not chosen, multi-GPU')
    parser.add_argument('--N', default=10,
                        help='Number of experiments.')
    parser.add_argument('--max_len', default=57,
                        help='Number of experiments.')
    parser.add_argument('--train', default='data/train.txt',
                        help='Train file')
    parser.add_argument('--test', default='data/test.txt',
                        help='Test file')
    parser.add_argument('--valid', default='data/valid.txt',
                        help='Validation file')
    parser.add_argument(
        '--directory',
        default='data/',
        help='Directory of the data (test and valid documents)')
    parser.add_argument(
        '--processed_directory',
        default='data/processed',
        help='Directory of the data (test and valid documents)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
#    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
#    parser.add_argument('--preserve-case', action='store_false', dest='lower')
#    parser.add_argument('--no-projection', action='store_false', dest='projection')
#    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--generate_data', action='store_true')
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument(
        '--vector_cache',
        type=str,
        default=os.path.join(
            os.getcwd(),
            '.vector_cache/input_vectors.pt'))
#    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os
    import errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise
