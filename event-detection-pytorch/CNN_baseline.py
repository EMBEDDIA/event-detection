# -*- coding: utf-8 -*-

"""
Nguyen, Thien Huu, and Ralph Grishman. 
"Event detection and domain adaptation with convolutional neural networks." 
Proceedings of the 53rd Annual Meeting of the Association for Computational 
Linguistics and the 7th International Joint Conference on Natural Language 
Processing (Volume 2: Short Papers). Vol. 2. 2015.
"""

import numpy as np
np.random.seed(42)  # for reproducibility
import warnings
import os
warnings.filterwarnings("ignore")
import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        # Restrict TensorFlow to only use the fourth GPU
#        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#
#        # Currently, memory growth needs to be the same across GPUs
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        # Memory growth must be set before GPUs have been initialized
#        print(e)
from ACE2005_evaluator import (load_gold, evaluate)
from utils import f2_score, recall_m, precision_m
import pickle as pkl
import gzip
import argparse
from datetime import datetime
from keras.utils import np_utils
from keras.optimizers import Adam

from keras.callbacks import (ModelCheckpoint, EarlyStopping,
                             TensorBoard, ReduceLROnPlateau)
from keras.utils import multi_gpu_model
import models
import utils
#from cyclical_learning_rate import CyclicLR
from sklearn.metrics import precision_recall_fscore_support


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
    

from keras.callbacks import Callback
logger = utils.get_logger('Event Detection - ')

MODELS_MAPPING = {'CNN2015': models.build_baseline_model_2015,
                  'CNN2018': models.build_baseline_model_2018,
                  'CNN2019': models.build_baseline_model_pytorch_2019,
                  'attention': models.get_bi}


INVERSE_MAPPING = {0: 'Other', 1: 'End-Position', 2: 'Attack',
                   3: 'Injure', 4: 'Transport', 5: 'Transfer-Ownership',
                   6: 'Declare-Bankruptcy', 7: 'Die', 8: 'Nominate',
                   9: 'Merge-Org', 10: 'Charge-Indict', 11: 'Transfer-Money',
                   12: 'Release-Parole', 13: 'Arrest-Jail',
                   14: 'Convict', 15: 'Meet', 16: 'Marry', 17: 'Sue',
                   18: 'Sentence', 19: 'Trial-Hearing', 20: 'Start-Position',
                   21: 'Phone-Write', 22: 'Elect', 23: 'Be-Born',
                   24: 'Demonstrate', 25: 'Start-Org', 26: 'End-Org',
                   27: 'Appeal', 28: 'Divorce', 29: 'Acquit',
                   30: 'Fine', 31: 'Execute', 32: 'Pardon', 33: 'Extradite'}


class TrainingHistory(Callback):

    def __init__(self, labels=[]):
        super(Callback, self).__init__()
        self.only_for_these_labels = labels

    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch_losses = []
        self.epoch_val_losses = []
        self.val_losses = []
        self.predictions = []
        self.epochs = []
        self.f1 = []
        self.i = 0
        self.save_every = 50

    def on_epoch_end(self, epoch, logs={}):

        print("\n--------- Epoch %d -----------" % (epoch+1))
        results_test = self.model.predict([windowTest, positionTest])
        results_valid = self.model.predict([windowValid, positionValid])

        p, r, f1, s = precision_recall_fscore_support(yValid, results_valid.argmax(1),
                                                      labels=self.only_for_these_labels,
                                                      average='micro')

        print('Valid p r f1 %.2f %.2f %.2f' % (np.average(p, weights=s)*100.0,
                                               np.average(r, weights=s)*100.0,
                                               np.average(f1, weights=s)*100.0))

        p, r, f1, s = precision_recall_fscore_support(yTest, results_test.argmax(1),
                                                      labels=self.only_for_these_labels,
                                                      average='micro')

        print('Test  p r f1 %.2f %.2f %.2f' % (np.average(p, weights=s)*100.0,
                                               np.average(r, weights=s)*100.0,
                                               np.average(f1, weights=s)*100.0))

        self.epochs.append(int(epoch))
        self.epoch_losses.append(logs.get('loss'))
        self.epoch_val_losses.append(logs.get('val_loss'))


def train_epoch(epoch, args, model, device, data_loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            
            
def run_experiment(gold_annots, embeddings, gold_test_file='results/ACE_gold2.tbf',
                   multi_gpu=True, write_results=False):

    only_for_these_labels = []
    target_names = []

    for targetLabel in range(0, max(yTest)+1):
        if not 'Other' in INVERSE_MAPPING[targetLabel]:
            only_for_these_labels.append(targetLabel)
            target_names.append(INVERSE_MAPPING[targetLabel])

    test_ids = []
    with open('data/test_split.txt', 'r') as f:
        for _id in f.readlines():
            test_ids.append(_id.replace('\n', '').split('/')[-1])

    if not os.path.exists('results/scores_baseline'):
        os.makedirs('results/scores_baseline')

    if not os.path.exists('data/test_split.txt'):
        print('I used to have the split ids here: data/test_split.txt')
    if not os.path.exists('results/ACE_05-test-orig.tbf'):
        print('I used to have the gold here: results/ACE_05-test.tbf')

    net = models.build_baseline_model_pytorch_2019(size_embeddings=embeddings.shape[1],
                                       window_length=windowTrain.shape[1],
                                       number_words=len(idx2word),
                                       number_positions=np.max(
                                           positionTrain)+1,
                                       number_labels=n_out,
                                       embeddings=embeddings)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    
    
    
    
    
    

    history = TrainingHistory(labels=only_for_these_labels)
    model = MODELS_MAPPING[model_name](size_embeddings=embeddings.shape[1],
                                       window_length=windowTrain.shape[1],
                                       number_words=len(idx2word),
                                       number_positions=np.max(
                                           positionTrain)+1,
                                       number_labels=n_out,
                                       embeddings=embeddings)

    model_json = model.to_json()

    if multi_gpu:
        gpus = len(utils.get_available_gpus())
        model = multi_gpu_model(model, gpus=gpus)

    weights_path = "weights/weights.trigger.baseline.hdf5"
    if not os.path.exists('weights'):
        os.mkdir('weights')

    checkpoint = ModelCheckpoint(weights_path,
                                 monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')

    early = EarlyStopping(monitor='val_loss', patience=3,
                          verbose=0, mode='min')

    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
#                  metrics=['accuracy'])
                  metrics=[f2_score, recall_m, precision_m])

    with open(pkl_dir+"/model_trigger_baseline_"+title + "_" + str(args.run) +
              ".json", "w") as json_file:
        json_file.write(model_json)

    model.fit([windowTrain, positionTrain],
              train_y_cat,
              validation_data=([windowValid, positionValid], valid_y_cat),
              batch_size=256, verbose=1, epochs=10,
              callbacks=[checkpoint, early, history])

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + '_'

    model.load_weights(weights_path)

    results_test = model.predict([windowTest, positionTest])
#    p, r, f1, s = precision_recall_fscore_support(yTest, results_test.argmax(1),
#                                                  labels=only_for_these_labels,
#                                                  average='micro')
#
#    print('Test  p r f1 %.2f %.2f %.2f' % (np.average(p, weights=s)*100.0,
#                                           np.average(r, weights=s)*100.0,
#                                           np.average(f1, weights=s)*100.0))

    if not os.path.exists('results/scores_baseline/'):
        os.makedirs('results/scores_baseline/')

    logger.info('Preparing results')

    results = []
    res = []
    for test_id in test_ids:

        if write_results:
            with open('results/scores_baseline/CNN_test_' + run_name +
                      str(args.run) + '.txt', 'a') as f_file:
                f_file.write('#BeginOfDocument ' + str(test_id) + '\n')

        index = 0

        for idx, val in enumerate(zip(windowTest, wholeTest, yTest, results_test.argmax(1), results_test)):
            sentence, whole_sentence, y, y_pred, y_pred_prob = val
            sent = []
            for id_ in sentence:
                sent.append(idx2word[id_])

            res_probs = y_pred_prob.argsort()[-2:][::-1]

            if str(detailsTest[idx][0]) in test_id:
                start, end = detailsTest[idx][2].split(',')
                if (detailsTest[idx][2], int(start), int(end), idx2word[sentence[15]], INVERSE_MAPPING[y_pred].lower()) not in res:
                    res.append((detailsTest[idx][2], int(start), int(
                        end), idx2word[sentence[15]], INVERSE_MAPPING[y_pred].lower()))
                    if 'Other' not in str(INVERSE_MAPPING[y_pred]):
                        #docId, id_event, start, end, text, event_type
                        start, end = str(detailsTest[idx][2]).split(',')
                        start, end = int(start), int(end)
                        results.append((str(detailsTest[idx][0]), 'EVENT'+str(index), start, end, str(
                            idx2word[sentence[15]]), str(INVERSE_MAPPING[y_pred])))
                        if write_results:
                            with open('results/scores_baseline/CNN_test_' + run_name +
                                      str(args.run) + '.txt', 'a') as f_file:
                                f_file.write('EB\t' + str(detailsTest[idx][0]) + '\t' +
                                             'EVENT'
                                             + '\t' + str(detailsTest[idx][2])
                                             + '\t' +
                                             str(idx2word[sentence[15]]) + '\t'
                                             + str(INVERSE_MAPPING[y_pred]) + '\t'
                                             + str(y_pred_prob[y_pred]
                                                   ) + '\tTrue' + '\t'
                                             + str(INVERSE_MAPPING[res_probs[-1]]) + ':\t'
                                             + str(y_pred_prob[res_probs[-1]]) + '\t'
                                             + '\n')

                        index += 1
        if write_results:
            with open('results/scores_baseline/CNN_test_' + run_name +
                      str(args.run) + '.txt', 'a') as f_file:
                f_file.write('#EndOfDocument\n')

    logger.info('Evaluation on test')
    if write_results:
        p, r, f = evaluate(gold_annots, 'results/scores_baseline/CNN_test_' + run_name +
                           str(args.run) + '.txt')
    else:
        p, r, f = evaluate(gold_annots, results)
    return p*100.0, r*100.0, f*100.0


if __name__ == '__main__':

    pkl_dir = 'data/processed'
    parser = argparse.ArgumentParser(
        description='CNN baseline')
    parser.add_argument('--embeddings', default='none', metavar='EMB',
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

    args = parser.parse_args()

    title = args.embeddings
    model_name = args.model
    gold_test_file = args.gold_test
    multi_gpu = bool(args.multi_gpu)
    write_results = bool(args.write_results)
    number_of_experiments = args.N

    assert os.path.exists(gold_test_file)

    idx2wordPklPath = pkl_dir + "/idx2word_"+title+".pkl.gz"
    f = gzip.open(idx2wordPklPath, 'rb')
    idx2word = pkl.load(f)
    f.close()
    
    print(len(idx2word))
    
    
    print("Load dataset")
    f = gzip.open(pkl_dir+'/triggers_'+title+'_train.pkl.gz', 'rb')
    _, detailsTrain, yTrain, windowTrain, positionTrain, \
        charTrain, wholeTrain, \
        allSentenceTrain, labelsSentenceTrain, wordsTrain = pkl.load(f)
    f.close()

    f = gzip.open(pkl_dir+'/triggers_'+title+'_test.pkl.gz', 'rb')
    _, detailsTest, yTest, windowTest, positionTest, charTest, \
        wholeTest, allSentenceTest, \
        labelsSentenceTest, wordsTest\
        = pkl.load(f)
    f.close()

    f = gzip.open(pkl_dir+'/triggers_'+title+'_valid.pkl.gz', 'rb')
    _, detailsValid, yValid, windowValid, positionValid, \
        charValid, wholeValid, \
        allSentenceValid, labelsSentenceValid, \
        wordsValid = pkl.load(f)
    f.close()

    n_out = len(INVERSE_MAPPING)

    train_y_cat = np_utils.to_categorical(yTrain, n_out)
    valid_y_cat = np_utils.to_categorical(yValid, n_out)

    embeddings = None
    if not args.embeddings == 'none':
        f = gzip.open(pkl_dir + '/embeddings_' + title + '.pkl.gz', 'rb')
        embeddings = pkl.load(f)
        embeddings = np.array(embeddings)
        f.close()
        print("Embeddings: ", embeddings.shape)

    print(len(idx2word), embeddings.shape)


    # Load only once the gold truth
    gold_annots = load_gold(args.gold_test)

    precisions, recalls, fs = [], [], []
    for idx in range(number_of_experiments):
        logger.info('Running experiment {}'.format(idx))
        p, r, f = run_experiment(gold_annots, embeddings,
                                 multi_gpu=multi_gpu,
                                 write_results=write_results)
        precisions.append(p)
        recalls.append(r)
        fs.append(f)

    precisions, recalls, fs = np.array(
        precisions), np.array(recalls), np.array(fs)
    print("Precision: %0.2f (+/- %0.4f)" %
          (precisions.mean(), precisions.std() / 2))
    print("Recall: %0.2f (+/- %0.4f)" % (recalls.mean(), recalls.std() / 2))
    print("F1: %0.2f (+/- %0.4f)" % (fs.mean(), fs.std() / 2))
