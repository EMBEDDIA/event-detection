#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:25:19 2018

@author: Emanuela Boros
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

import warnings
warnings.filterwarnings("ignore")

from ACE2005_evaluator import (load_gold, evaluate)

import pickle as pkl
import gzip
import os
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers import (Conv1D, Input, Dropout,
                          GlobalMaxPooling1D)
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.layers.merge import concatenate
import keras
from keras.callbacks import ModelCheckpoint
import argparse
import utils
import datetime
from sklearn.metrics import precision_recall_fscore_support
from utils import f2_score, recall_m, precision_m
from keras.optimizers import Adam
from enum import Enum
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)

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

logger = utils.get_logger('Event Detection - CNN + char embeddings')
    
def create_model(type_model, size_embeddings):
    ngram_filters = [1, 2, 3]
    filters = [300]*20
    graph_in = Input(shape=(None, size_embeddings+50))
    convs = []
    for kernel_size, filter_length in zip(ngram_filters, filters):
        conv = Conv1D(filters=filter_length,
                      kernel_size=kernel_size,
                      activation='relu',
                      strides=1,
                      padding='same',
                      kernel_initializer=keras.initializers.Orthogonal(gain=1.0, seed=42))(graph_in)
        convs.append(conv)

    x = concatenate(convs)
    x = GlobalMaxPooling1D()(x)

    convolutions = Model(inputs=graph_in, outputs=x, name='convolutions')

    char_graph_in = Input(shape=(1024, 300))
    convolution_output = []
    # 10, 9,
    for kernel_size, filter_length in zip([10, 9, 8, 7, 6, 5, 4, 3, 2], [300]*10):
        x = Conv1D(filters=filter_length,
                   kernel_size=kernel_size,
                   activation='relu',
                   kernel_initializer=keras.initializers.Orthogonal(
                       gain=1.0, seed=42),
                   name='Conv1D_{}_{}'.format(kernel_size, filter_length))(char_graph_in)

        pool = GlobalMaxPooling1D(
            name='MaxPoolingOverTime_{}_{}'.format(kernel_size, filter_length))(x)
        convolution_output.append(pool)

    char_x = concatenate(convolution_output)

    char_convolutions = Model(inputs=char_graph_in,
                              outputs=char_x, name='char_convolutions')

    MAX_SEQUENCE_LENGTH = sentenceTrain.shape[1]
    MAX_SENTENCE_LENGTH = wholeTrain.shape[1]

    nb_words = embeddings.shape[0]

    embedding_layer = Embedding(nb_words,
                                embeddings.shape[1],
                                weights=[embeddings],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False, name='embedded_words')

    embedding_distance_layer = Embedding(max_position,
                                         50,
                                         input_length=MAX_SEQUENCE_LENGTH,
                                         trainable=True,
                                         name='embedded_distances')

    char_embedding_layer = Embedding(len(char2idx),
                                     300,
                                     input_length=charTrain.shape[1],
                                     trainable=True, name='embedded_char')


    sequence_sent_input = Input(shape=(MAX_SEQUENCE_LENGTH,),
                                dtype='int32',
                                name='sequence_words')
    embedded_sent = embedding_layer(sequence_sent_input)
    embedded_sent = Dropout(0.3)(embedded_sent)

    sequence_dist_input = Input(shape=(MAX_SEQUENCE_LENGTH,),
                                dtype='int32',
                                name='sequence_distances')
    embedded_dist = embedding_distance_layer(sequence_dist_input)

    sequence_char_input = Input(
        shape=(charTrain.shape[1], ), dtype='int32', name='sequence_characters')
    embedded_char = char_embedding_layer(sequence_char_input)
    conv_char = char_convolutions(embedded_char)

    auxiliary_output = Dense(
        n_out, activation='softmax', name='aux_output')(conv_char)

    merged = concatenate([embedded_sent, embedded_dist])
    merged = convolutions(merged)
    
    if type_model.lower() == 'joint':

#        merged = keras.layers.concatenate([keras.layers.Dense(100, activation="tanh")(conv_char),
#                                           keras.layers.Dense(100, activation="tanh")(conv_words)])
        merged = concatenate([conv_char, merged])
        merged = Dropout(0.5)(merged)
    
        main_output = Dense(n_out, activation='softmax',
                            name='main_output')(merged)
        model = Model(inputs=[sequence_sent_input, sequence_dist_input, sequence_char_input],
                      outputs=[main_output])
    else:
        merged = Dropout(0.5)(merged)
    
        main_output = Dense(n_out, activation='softmax',
                            name='main_output')(merged)
        model = Model(inputs=[sequence_sent_input, sequence_dist_input, sequence_char_input],
                      outputs=[main_output, auxiliary_output])

    model.summary()

    return model


def run_experiment(gold_annots, embeddings, gold_test_file='results/ACE_gold2.tbf',
                   multi_gpu=True, write_results=False, type_model='late'):

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
    valid_ids = []
    with open('data/valid_split.txt', 'r') as f:
        for _id in f.readlines():
            valid_ids.append(_id.replace('\n', '').split('/')[-1])

    logger.info('Training baseline +  char model')
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    
    if type_model.lower() == 'late':
        model.compile(optimizer=optimizer,
                      metrics=[f2_score, recall_m, precision_m],
                      loss={'main_output': 'categorical_crossentropy',
                            'aux_output': 'categorical_crossentropy'},
                      loss_weights={'main_output': .3, 'aux_output': 1.})
    else:
        model.compile(optimizer=optimizer,
                      metrics=[f2_score, recall_m, precision_m],
                      loss={'main_output': 'categorical_crossentropy'})
        
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  patience=2, verbose=1)

    weights_path = 'weights_char/weights.trigger.baseline.char.hdf5'

    checkpoint = ModelCheckpoint(weights_path,
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    with open(pkl_dir+"/model_trigger_NORMAL_"+title+".json", "w") as json_file:
        json_file.write(model_json)
    early = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2, verbose=0, mode='auto')


    best_val = 0.0
    best_model = None
    
    for epoch in range(10):
        
        if type_model.lower() == 'late':
            model.fit([sentenceTrain, positionTrain, charTrain],
                      [train_y_cat, train_y_cat],
                      validation_data=([sentenceValid, positionValid, charValid], [
                                       valid_y_cat, valid_y_cat]),
                      batch_size=batch_size, verbose=1, epochs=1, callbacks=[early, reduce_lr])
        else:
            model.fit([sentenceTrain, positionTrain, charTrain],
                      [train_y_cat],
                      validation_data=([sentenceValid, positionValid, charValid], [
                                       valid_y_cat]),
                      batch_size=batch_size, verbose=1, epochs=1, callbacks=[early, reduce_lr])
                  
        print("\n--------- Epoch %d -----------" % (epoch+1))
        results_test = model.predict(
            [sentenceTest, positionTest, charTest])
        
        if type_model.lower() == 'joint':
            results_test1 = results_test
            results_test2 = results_test
        
        else:
            results_test1 = results_test[0]
            results_test2 = results_test[1]

        results_valid = model.predict(
            [sentenceValid, positionValid, charValid])

        if type_model.lower() == 'joint':
            results_valid1 = results_valid
            results_valid2 = results_valid
        
        else:
            results_valid1 = results_valid[0]
            results_valid2 = results_valid[1]


        p, r, f1, s = precision_recall_fscore_support(yValid, results_valid1.argmax(1),
                                                      labels=only_for_these_labels, average='micro')
#            print('valid p r f1 %.2f %.2f %.2f' % (np.average(p, weights=s)*100.0, np.average(r, weights=s)*100.0,
#                                                   np.average(f1, weights=s)*100.0))
#
#
        p, r, f1_val, s = precision_recall_fscore_support(yValid, results_valid2.argmax(1),
                                                          labels=only_for_these_labels,  average='micro')
#            print('valid p r f1 %.2f %.2f %.2f' % (np.average(p, weights=s)*100.0, np.average(r, weights=s)*100.0,
#                                                   np.average(f1_val, weights=s)*100.0))

        if f1_val > best_val:
            best_val = f1_val
            best_model = model
            
  
    results_test = best_model.predict(
        [sentenceTest, positionTest, charTest])
    
    if type_model.lower() == 'late':
        results_test1, results_test2 = results_test[0], results_test[1]
    else:
        results_test1, results_test2 = results_test, results_test
        
# CHARACTER
    res = []
    results = []
    results_char = []
    
    if type_model.lower() == 'late':
        for test_id in test_ids:
            with open('results/scores_just_char/CNN_test_epoch' + str(epoch+1) + '_' +
                      str(args.run) + '.txt', 'a') as f_file:
                f_file.write('#BeginOfDocument ' + str(test_id) + '\n')
    
            index = 0
            for idx, val in enumerate(zip(sentenceTest, wholeTest, yTest, results_test2.argmax(1), results_test2)):
                sentence, whole_sentence, y, y_pred, y_pred_prob = val
                sent = []
                for id_ in sentence:
                    sent.append(idx2word[id_])
    
                res_probs = y_pred_prob.argsort()[-2:][::-1]
    
                with open('results/scores_just_char/CNN_test_epoch' + str(epoch+1) + '_' +
                          str(args.run) + '.txt', 'a') as f_file:
                    if str(detailsTest[idx][0]) in test_id:
                        start, end = detailsTest[idx][2].split(',')
                        if (detailsTest[idx][2], int(start), int(end), idx2word[sentence[15]], INVERSE_MAPPING[y_pred].lower()) not in res:
                            res.append((detailsTest[idx][2], int(start), int(
                                end), idx2word[sentence[15]], INVERSE_MAPPING[y_pred].lower()))
                            if 'Other' not in str(INVERSE_MAPPING[y_pred]):
                                start, end = str(detailsTest[idx][2]).split(',')
                                start, end = int(start), int(end)
                                results_char.append((str(detailsTest[idx][0]), 'EVENT'+str(index), start, end, str(
                                    idx2word[sentence[15]]), str(INVERSE_MAPPING[y_pred])))
                                if write_results:
                                    f_file.write('EB\t' + str(detailsTest[idx][0]) + '\t' +
                                                 str(detailsTest[idx][1])
                                                 + '\t' +
                                                 str(detailsTest[idx][2])
                                                 + '\t' +
                                                 str(idx2word[sentence[15]]) + '\t'
                                                 + str(INVERSE_MAPPING[y_pred]) + '\t'
                                                 + str(y_pred_prob[y_pred]) + '\tTrue' + '\t'
                                                 + str(INVERSE_MAPPING[res_probs[-1]]) + ':\t'
                                                 + str(y_pred_prob[res_probs[-1]]) + '\t'
                                                 + '\n')
    
                                index += 1
    
            with open('results/scores_just_char/CNN_test_epoch' + str(epoch+1) + '_' +
                      str(args.run) + '.txt', 'a') as f_file:
                f_file.write('#EndOfDocument\n')

        print('TEST ONLY CHARACTER')
        p_char, r_char, f_char = evaluate('results/ACE_05-test-orig.tbf', 'results/scores_just_char/CNN_test_epoch' + str(epoch+1) + '_' +
                                          str(args.run) + '.txt')
    
        logger.info('Evaluation on test')
        if write_results:
            p_char, r_char, f_char = evaluate(gold_annots, 'results/scores_just_char/CNN_test_epoch' + str(epoch+1) + '_' +
                                              str(args.run) + '.txt')
    
        else:
            p_char, r_char, f_char = evaluate(gold_annots, results_char)
            


        precisions_char.append(p_char)
        recalls_char.append(r_char)
        fs_char.append(f_char)


    res = []
    run_name = str(datetime.datetime.now()).replace(
        ' ', '-').split(':')[0] + '_'
    for test_id in test_ids:
        with open('results/scores_char/CNN_test_epoch' + str(epoch+1) + '_' + run_name +
                  str(args.run) + '.txt', 'a') as f_file:
            f_file.write('#BeginOfDocument ' + str(test_id) + '\n')

        index = 0
        for idx, val in enumerate(zip(sentenceTest, wholeTest, yTest, results_test1.argmax(1), results_test1, results_test2.argmax(1))):
            sentence, whole_sentence, y, y_pred1, y_pred_prob1, y_pred2 = val
            sent = []
            for id_ in sentence:
                sent.append(idx2word[id_])

            res_probs = y_pred_prob1.argsort()[-2:][::-1]

            with open('results/scores_char/CNN_test_epoch' + str(epoch+1) + '_' + run_name +
                      str(args.run) + '.txt', 'a') as f_file:
                if str(detailsTest[idx][0]) in test_id:

                    start, end = detailsTest[idx][2].split(',')
                    if (detailsTest[idx][2], int(start), int(end)) not in res:
                        res.append(
                            (detailsTest[idx][2], int(start), int(end)))
                        if 'Other' not in str(INVERSE_MAPPING[y_pred1]):
                                if 'Other' not in str(INVERSE_MAPPING[y_pred2]):
                                  
                                    start, end = str(detailsTest[idx][2]).split(',')
                                    start, end = int(start), int(end)
                                    results.append((str(detailsTest[idx][0]), 'EVENT'+str(index), start, end, str(
                                        idx2word[sentence[15]]), str(INVERSE_MAPPING[y_pred2])))

                                    if write_results:
                                        f_file.write('EB\t' + str(detailsTest[idx][0]) + '\t' +
                                                     str(detailsTest[idx][1])
                                                     + '\t' +
                                                     str(detailsTest[idx][2])
                                                     + '\t' +
                                                     str(idx2word[sentence[15]]) + '\t'
                                                     + str(INVERSE_MAPPING[y_pred2]) + '\t'
                                                     + str('MODIFIED_BY_CHAR') + '\tTrue'
                                                     + '\n')
                                else:
                                    start, end = str(detailsTest[idx][2]).split(',')
                                    start, end = int(start), int(end)
                                    results.append((str(detailsTest[idx][0]), 'EVENT'+str(index), start, end, str(
                                        idx2word[sentence[15]]), str(INVERSE_MAPPING[y_pred1])))
                                    if write_results:
                                        f_file.write('EB\t' + str(detailsTest[idx][0]) + '\t' +
                                                     str(detailsTest[idx][1])
                                                     + '\t' +
                                                     str(detailsTest[idx][2])
                                                     + '\t' +
                                                     str(idx2word[sentence[15]]) + '\t'
                                                     + str(INVERSE_MAPPING[y_pred1]) + '\t'
                                                     + 'ONLY_BASELINE' + '\tTrue' + '\t'
                                                     + '\n')

                                    index += 1

                        else:
                            if 'Other' not in str(INVERSE_MAPPING[y_pred2]):
                                start, end = str(detailsTest[idx][2]).split(',')
                                start, end = int(start), int(end)
                                results.append((str(detailsTest[idx][0]), 'EVENT'+str(index), start, end, str(
                                    idx2word[sentence[15]]), str(INVERSE_MAPPING[y_pred2])))
                                if write_results:
                                    f_file.write('EB\t' + str(detailsTest[idx][0]) + '\t' +
                                                 str(detailsTest[idx][1])
                                                 + '\t' +
                                                 str(detailsTest[idx][2])
                                                 + '\t' +
                                                 str(idx2word[sentence[15]]) + '\t'
                                                 + str(INVERSE_MAPPING[y_pred2]) + '\t'
                                                 + str('ONLY_CHAR') +
                                                 '\tTrue'
                                                 + '\n')

                                    index += 1
    if write_results:
        with open('results/scores_char/CNN_test_epoch' + str(epoch+1) + '_' + run_name +
                  str(args.run) + '.txt', 'a') as f_file:
            f_file.write('#EndOfDocument\n')

#        print('TEST')
#        p, r, f = evaluate('results/ACE_05-test-orig.tbf', 'results/scores_char/CNN_test_epoch' + str(epoch+1) + '_' + run_name +
#                           str(args.run) + '.txt')
        
    logger.info('Results on test:')
    if write_results:
        p, r, f = evaluate(gold_annots, 'results/scores_char/CNN_test_epoch' + str(epoch+1) + '_' + run_name +
                           str(args.run) + '.txt')
    else:
        p, r, f = evaluate(gold_annots, results)
    
    if type_model.lower() == 'joint':
        return p, r, f
    else:
        return p, r, f, p_char, r_char, f_char

  
if __name__ == '__main__':

    pkl_dir = 'data/processed'
    parser = argparse.ArgumentParser(
        description='CNN basic')
    parser.add_argument('-e', '--embeddings', default=None, metavar='EMB',
                        help='embeddings type: google, deps, gigaword, numberbatch, none')
    parser.add_argument('-r', '--run', default=1,
                        metavar='RUN', help='run number/details')
    parser.add_argument('-gt', '--gold_test', default='results/ACE_gold2.tbf',
                        help='test gold file')
    parser.add_argument('-gv', '--gold_valid', default='results/ACE_valid.tbf',
                        help='valid gold file')
    parser.add_argument('--write_results', default=0,
                        help='0 - no, 1 - yes')
    parser.add_argument('--model',
                        default='LATE',
                        nargs='?',
                        choices=['JOINT', 'LATE'],
                        help='list models (default: %(default)s)')
    parser.add_argument('--multi-gpu', default=1,
                        help='If GPU is not chosen, multi-GPU')
    parser.add_argument('--N', default=5,
                        help='Number of experiments.')
    parser.add_argument('--batch_size', default=128,
                        help='Number of batches.')
    parser.add_argument('--size_embeddings', default=300, #3072 BERT
                        help='Number of batches.')

    args = parser.parse_args()
    title = args.embeddings
    type_model = args.model
    number_of_experiments = args.N
    batch_size = args.batch_size
    write_results = bool(args.write_results)
    multi_gpu = bool(args.multi_gpu)
    size_embeddings = int(args.size_embeddings)

    idx2wordPklPath = pkl_dir + "/idx2word_" + title + ".pkl.gz"
    f = gzip.open(idx2wordPklPath, 'rb')
    idx2word = pkl.load(f)
    f.close()

    char2idxPklPath = pkl_dir+"/char2idx_"+args.embeddings+".pkl.gz"
    f = gzip.open(char2idxPklPath, 'rb')
    char2idx = pkl.load(f)
    f.close()

    logger.info("Loading dataset")
    f = gzip.open(pkl_dir+'/triggers_'+title+'_train.pkl.gz', 'rb')
    _, detailsTrain, yTrain, sentenceTrain, positionTrain, \
        charTrain, wholeTrain, \
        allSentenceTrain, labelsSentenceTrain, wordsTrain = pkl.load(f)
    f.close()

    f = gzip.open(pkl_dir+'/triggers_'+title+'_test.pkl.gz', 'rb')
    _, detailsTest, yTest, sentenceTest, positionTest, charTest, \
        wholeTest, allSentenceTest, \
        labelsSentenceTest, wordsTest\
        = pkl.load(f)
    f.close()

    f = gzip.open(pkl_dir+'/triggers_'+title+'_valid.pkl.gz', 'rb')
    _, detailsValid, yValid, sentenceValid, positionValid, \
        charValid, wholeValid, \
        allSentenceValid, labelsSentenceValid, \
        wordsValid = pkl.load(f)
    f.close()
    
    f = gzip.open(pkl_dir+'/embeddings_'+title+'.pkl.gz', 'rb')
    embeddings = pkl.load(f)
    embeddings = np.array(embeddings)
    f.close()

    print("Embeddings: ", embeddings.shape)

    gold_annots = load_gold(args.gold_test)

    max_position = np.max(positionTrain)+1

    labels_mapping = {}
    for no, label in INVERSE_MAPPING.items():
        labels_mapping[label] = no

    n_out = 34

    train_y_cat = np_utils.to_categorical(yTrain, n_out)
    valid_y_cat = np_utils.to_categorical(yValid, n_out)

    #TODO
#    def read_lexicon(filename):
#        lexicon = {}
#        for line in open(filename, 'r'):
#            words = line.strip().split()
#            lexicon[words[0]] = [word.lower() for word in words[1:]]
#        return lexicon

#    adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)

    if not os.path.exists('results/scores_basic'):
        os.makedirs('results/scores_basic')
    if not os.path.exists('results/scores_just_char'):
        os.makedirs('results/scores_just_char')
    if not os.path.exists('results/scores_char'):
        os.makedirs('results/scores_char')

    if not os.path.exists('data/test_split.txt'):
        print('I used to have the split ids here: data/test_split.txt')
    if not os.path.exists('results/ACE_05-test-orig.tbf'):
        print('I used to have the gold here: results/ACE_05-test.tbf')

    model = create_model(type_model, size_embeddings)
    if multi_gpu:
        gpus = len(utils.get_available_gpus())
        if gpus > 1:
            logger.info('Running on ' + str(gpus) + ' gpus')
            model = multi_gpu_model(model, gpus=gpus)

    model_json = model.to_json()

    precisions, recalls, fs = [], [], []
    precisions_char, recalls_char, fs_char = [], [], []
    
    for idx in range(number_of_experiments):
        logger.info('Running experiment {}'.format(idx))
        
        if not type_model.lower() == 'joint':
            p, r, f, p_char, r_char, f_char = run_experiment(gold_annots, embeddings,
                                     multi_gpu=multi_gpu,
                                     write_results=write_results,
                                     type_model=type_model)
            
            precisions_char.append(p_char)
            recalls_char.append(r_char)
            fs_char.append(f_char)
            
        else:
            p, r, f = run_experiment(gold_annots, embeddings,
                                     multi_gpu=multi_gpu,
                                     write_results=write_results,
                                     type_model=type_model)
        precisions.append(p)
        recalls.append(r)
        fs.append(f)



    precisions, recalls, fs = np.array(
        precisions), np.array(recalls), np.array(fs)
    print("Precision: %0.2f (+/- %0.4f)" %
          (precisions.mean(), precisions.std() / 2))
    print("Recall: %0.2f (+/- %0.4f)" % (recalls.mean(), recalls.std() / 2))
    print("F1: %0.2f (+/- %0.4f)" % (fs.mean(), fs.std() / 2))
    
    if not type_model.lower() == 'joint':
        precisions_char, recalls_char, fs_char = np.array(
            precisions_char), np.array(recalls_char), np.array(fs_char)
        print("Precision Char: %0.2f (+/- %0.4f)" %
              (precisions_char.mean(), precisions_char.std() / 2))
        print("Recall Char: %0.2f (+/- %0.4f)" %
              (recalls_char.mean(), recalls_char.std() / 2))
        print("F1 Char: %0.2f (+/- %0.4f)" % (fs_char.mean(), fs_char.std() / 2))
