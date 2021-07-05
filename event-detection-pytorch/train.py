# -*- coding: utf-8 -*-

import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
from model import (ACE2019, CNN_with_words,
                   CNN_with_attention,
                   CNN_with_chars,
                   Attention_Net)
from extract_data import prepare_dataset
from torch.autograd import Variable
#from util import get_args, makedirs
from utils import EarlyStopping
import json
#from ACE2005_evaluator import (load_gold, evaluate)
#from utils import f2_score, recall_m, precision_m
import pickle as pkl
import gzip
from utils import get_args, makedirs, get_logger
logger = get_logger('Event Detection - training')

#INVERSE_MAPPING = {0: 'Other', 1: 'Event'}

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


LANGUAGE_MAPPING = {'en': 0, 'fr': 1, 'el': 2, 'pl': 3, 'zh': 4, 'ru': 5}
INVERSE_LANGUAGE_MAPPING = {v: k for k, v in LANGUAGE_MAPPING.items()}


args = get_args()
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
else:
    device = torch.device('cpu')

iterations = 0
start_time = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)


def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))


def write_results_to_file(
        ids,
        run_name,
        epoch,
        tokens,
        details,
        all_y_test,
        all_scores_test,
        languages_test,
        type_eval='test',
        directory=None):
    print(type_eval)
    results_json = {}
    write_results = True
    if write_results:
        print('Preparing results')
#        results = []
        res = []

        for test_id in ids:
            annotations = set()
            annotations_with_positions = []
            print('-' * 50)
            print(test_id)
#            if write_results:
#                with open('results/scores_baseline/CNN_daniel_' + type_eval + '_' + run_name +
#                          str(args.run) + '_epoch_' + str(epoch) +  '.txt', 'a') as f_file:
#                    f_file.write('#BeginOfDocument ' + str(test_id) + '\n')
#
            index = 0
            languages = []
            for idx, val in enumerate(
                    zip(tokens, details, all_y_test, all_scores_test, languages_test)):
                sentence, detail, y, y_pred, language = val
#                sent = []
#                for id_ in sentence:
#                    sent.append(idx2word[id_.item()])

                if str(detail[0]) in test_id:
                    start, end = detail[2].split(',')
                    languages.append(language)

                    if (detail[2],
                        int(start),
                        int(end),
                        idx2word[sentence[15].item()],
                            INVERSE_MAPPING[y_pred.item()].lower()) not in res:
                        res.append((detail[2], int(start), int(
                            end), idx2word[sentence[15].item()], INVERSE_MAPPING[y_pred.item()].lower()))
                        if 'Other' not in str(INVERSE_MAPPING[y_pred.item()]):
                            print(str(idx2word[sentence[15].item()]), str(
                                INVERSE_MAPPING[y_pred.item()]), str(INVERSE_MAPPING[y.item()]))
                            #docId, id_event, start, end, text, event_type
#                                start, end = str(detail[2]).split(',')
#                                start, end = int(start), int(end)
#                                results.append((str(detail[0]), 'EVENT' + str(index), start, end, str(
#                                    idx2word[sentence[15].item()]), str(INVERSE_MAPPING[y_pred.item()])))
#                            if write_results:
#                                with open('results/scores_baseline/CNN_daniel_' + type_eval + '_' + run_name +
#                          str(args.run) + '_epoch_' + str(epoch) +  '.txt', 'a') as f_file:
#                                    f_file.write('EB\t' + str(detail[0]) + '\t' +
#                                                 'EVENT'
#                                                 + '\t' + str(detail[2])
#                                                 + '\t' +
#                                                 str(idx2word[sentence[15].item()]) + '\t'
#                                                 + str(INVERSE_MAPPING[y_pred.item()]) + '\t'
#                                                 + '\n')
                            annotations.add(str(idx2word[sentence[15].item()]))
                            annotations_with_positions.append(
                                (str(idx2word[sentence[15].item()]), detail[2]))
                            index += 1
#            if write_results:
#                with open('results/scores_baseline/CNN_daniel_' + type_eval + '_' + run_name +
#                          str(args.run) + '_epoch_' + str(epoch) +  '.txt', 'a') as f_file:
#                    f_file.write('#EndOfDocument\n')
#

            results_json[test_id] = {}
            annotations = list(annotations)

#            already_annotated = []

            if len(annotations_with_positions) > 0:
                new_annotations = []
                for idx_first, annotation_first in enumerate(
                        annotations_with_positions):
                    new_annotation = []

                    for idx_second, annotation_second in enumerate(
                            annotations_with_positions):
                        if annotation_first[0] != annotation_second[0]:
                            if annotation_first[1] == annotation_second[1]:
                                if annotation_second[0] + ' ' + \
                                        annotation_first[0] not in new_annotations:
                                    new_annotation.append(
                                        annotation_first[0] + ' ' + annotation_second[0])
                                    annotations_with_positions.remove(
                                        annotation_second)
                                    annotations_with_positions.remove(
                                        annotation_first)

                    if len(new_annotation) > 0:
                        new_annotations.append(new_annotation)
                if len(new_annotations) > 0:
                    print('---', new_annotations)
                    for new_annotation in new_annotations:
                        annotations.append(new_annotation[0])

            if len(annotations) == 0:
                annotations = ['N', 'N', 'N']
            if len(annotations) == 1:
                annotations.append('N')
                annotations.append('N')
            if len(annotations) == 2:
                annotations.append('N')
            results_json[test_id]['annotations'] = [annotations]
            try:
                results_json[test_id]['language'] = languages[0]
            except BaseException:
                print(test_id)
        print('write json')
#        with open('results/scores_baseline/CNN_daniel_' + type_eval + '_' + run_name +
#                  str(args.run) + '_epoch_' + str(epoch) + '.txt', 'a') as f_file:
#            f_file.write('#EndOfDocument\n')

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, 'CNN_' + type_eval + '_' + run_name +
                           str(args.run) + '_epoch_' + str(epoch) + '.json'), "w") as write_file:
        json.dump(
            results_json,
            write_file,
            ensure_ascii=True,
            indent=4,
            sort_keys=True)
#    json_data = json.dumps(results_json)


def predict_one_by_one(model, data):
    n_test_correct, test_loss = 0, 0
    test_loss_final = 0

    positions = torch.from_numpy(
        np.array([x['position_matrix'] for x in data]))
    details = np.array([x['details_event'] for x in data])
    labels = torch.from_numpy(
        np.array([x['labels'] for x in data]))
    tokens = torch.from_numpy(
        np.array([x['token_matrix'] for x in data]))
    languages = torch.from_numpy(
        np.array([x['language'] for x in data]))
    chars = torch.from_numpy(
        np.array([x['char_matrix'] for x in data]))

    with torch.no_grad():
        all_y_test, all_scores_test, all_languages_test = [], [], []
        for test_batch_idx, test_data_element in enumerate(
                zip(tokens, chars, details, positions, labels, languages)):

            words_test, chars_test, details_test, positions_test, y_test, language_test = test_data_element

            tokens_test = Variable(torch.LongTensor(words_test)).cuda()
            chars_test = Variable(torch.LongTensor(chars_test.long())).cuda()
            labels_test = Variable(torch.LongTensor([y_test])).cuda()
            positions_test = Variable(
                torch.LongTensor(positions_test.long())).cuda()

            scores_test = model(tokens_test.unsqueeze(
                0), positions_test.unsqueeze(0))
#            scores_test = model(chars_test.unsqueeze(0))

            n_test_correct += (torch.max(scores_test, 1)
                               [1].view(labels_test.size()) == labels_test).sum().item()

            all_y_test.append(y_test.cpu())
            all_scores_test.append(torch.max(scores_test, 1)[
                                   1].view(labels_test.size()).cpu())

            test_loss = criterion(scores_test, labels_test)
            test_loss_final += test_loss.item()
            all_languages_test.append(
                INVERSE_LANGUAGE_MAPPING[language_test.item()])

    all_languages_test = np.array(all_languages_test)
    all_y_test = np.array(all_y_test)
    all_scores_test = torch.cat(all_scores_test, dim=-1)
    return all_y_test, all_scores_test, all_languages_test, details, tokens


def predict_batch(model, loader):
    n_dev_correct, dev_loss = 0, 0
    valid_loss = 0
    with torch.no_grad():
        all_y_valid, all_scores_valid, languages_valid = [], [], []
        for valid_batch_idx, valid_batch in enumerate(loader):

            words_valid, chars_valid, positions_valid, y_valid, language_valid = valid_batch

            tokens_valid = Variable(torch.LongTensor(words_valid)).cuda()
            chars_valid = Variable(torch.LongTensor(chars_valid.long())).cuda()
            labels_valid = Variable(
                torch.LongTensor(y_valid.long())).cuda()
            positions_valid = Variable(
                torch.LongTensor(positions_valid.long())).cuda()

            scores_valid = model(tokens_valid, positions_valid)
#            scores_valid = model(chars_valid)
            # scores, labels.float().unsqueeze(1))
#            import pdb;pdb.set_trace()

            #
            n_dev_correct += (torch.max(scores_valid, 1)
                              [1].view(labels_valid.size()) == labels_valid).sum().item()

            all_y_valid.append(y_valid.cpu())
            languages_valid.append(language_valid.cpu())
            all_scores_valid.append(torch.max(scores_valid, 1)[
                                    1].view(labels_valid.size()).cpu())

            dev_loss = criterion(scores_valid, labels_valid)
            valid_loss += dev_loss.item()

    languages_valid = [INVERSE_LANGUAGE_MAPPING[x.item()]
                       for x in torch.cat(languages_valid, dim=-1)]
    all_y_valid = torch.cat(all_y_valid, dim=-1)
    all_scores_valid = torch.cat(all_scores_valid, dim=-1)

    return all_y_valid, all_scores_valid, languages_valid, valid_loss, n_dev_correct


def train_words(
        criterion,
        optimizer,
        loader_train,
        loader_valid,
        loader_test,
        valid_data,
        test_data,
        number_run,
        directory):
    iterations = 0

    early_stopping = EarlyStopping(patience=3, verbose=True)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
            nn.init.orthogonal_(m.bias.data)

    weights_init(model)

    best_model = model
    model.train(True)
    lowest_loss = 100000.0

    print(header)
    for epoch in range(args.epochs):
        #        loader_train.init_epoch()
        n_correct, n_total = 0, 0

        all_y_train, all_scores_train = [], []

        running_loss = 0

        for batch_idx, batch in tqdm(
                enumerate(loader_train), total=len(loader_train)):
            words_input, chars_input, positions, y, language = batch
            # switch model to training mode, clear gradient accumulators
            if early_stopping.early_stop:
                print(early_stopping.early_stop)
                print("Early stopping")
                break

            optimizer.zero_grad()

            iterations += 1

#            print('y', y.shape)

            tokens = Variable(torch.LongTensor(words_input)).cuda()
            chars = Variable(torch.LongTensor(chars_input.long())).cuda()
            positions = Variable(torch.LongTensor(positions.long())).cuda()
            labels = Variable(torch.LongTensor(y.long())).cuda()

            # forward pass
            scores = model(tokens, positions)
#            print('scores', scores.shape)
#            scores = model(chars)

            words = []
            for idx in words_input[0]:
                idx = idx.item()
                words.append(idx2word[idx])

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(scores, 1)
                          [1].view(labels.size()) == labels).float().sum().item()

            n_total += len(words_input)

            # calculate loss of the network output with respect to training labels
            # print(scores.type(), labels.type())
            #import pdb;pdb.set_trace()
            #loss = criterion(scores, labels.float().unsqueeze(1))
            loss = criterion(scores, labels)
            running_loss += loss.item()

            # backpropagate and update optimizer learning rate
            loss.backward(retain_graph=True)
#            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            all_y_train.append(y.cpu())
            all_scores_train.append(torch.max(scores, 1)[
                                    1].view(labels.size()).cpu())

        # switch model to evaluation mode
        model.eval()

        logger.info('Evaluation')
#
#        all_y_train = torch.cat(all_y_train, dim=-1)
#        all_scores_train = torch.cat(all_scores_train, dim=-1)
#
#        p, r, f1, s = precision_recall_fscore_support(all_y_train,
#                                                      all_scores_train,
#                                                      labels=list(
#                                                          INVERSE_MAPPING.keys() - [0]),
#                                                      average='micro')
#
#        print('train p r f1 %.2f %.2f %.2f' % (np.average(p, weights=s)*100.0,
#                                               np.average(r, weights=s)*100.0,
#                                               np.average(f1, weights=s)*100.0))
#
        train_acc = 100. * n_correct / n_total

        all_y_valid, all_scores_valid, languages_valid, valid_loss, n_dev_correct = predict_batch(
            model, loader_valid)
        p_val, r_val, f1_val, s_val = precision_recall_fscore_support(
            all_y_valid, all_scores_valid, labels=list(
                INVERSE_MAPPING.keys() - [0]), average='micro')

        print(
            'valid p r f1 %.2f %.2f %.2f' %
            (np.average(
                p_val,
                weights=s_val) *
                100.0,
                np.average(
                r_val,
                weights=s_val) *
                100.0,
                np.average(
                f1_val,
                weights=s_val) *
                100.0))

        dev_acc = 100. * n_dev_correct / len(all_scores_valid)

        print(dev_log_template.format(time.time() - start_time,
                                      epoch,
                                      iterations,
                                      1 + batch_idx,
                                      len(loader_train),
                                      100. * (1 + batch_idx) / len(loader_train),
                                      running_loss / len(loader_train),
                                      valid_loss / len(loader_valid),
                                      train_acc,
                                      dev_acc))

        all_y_test, all_scores_test, languages_test, test_loss, n_test_correct = predict_batch(
            model, loader_test)
        p_test, r_test, f1_test, s_test = precision_recall_fscore_support(
            all_y_test, all_scores_test, labels=list(
                INVERSE_MAPPING.keys() - [0]), average='micro')

        print(
            'test p r f1 %.2f %.2f %.2f' %
            (np.average(
                p_test,
                weights=s_test) *
                100.0,
                np.average(
                r_test,
                weights=s_test) *
                100.0,
                np.average(
                f1_test,
                weights=s_test) *
                100.0))

        if valid_loss / len(loader_valid) < lowest_loss:
            lowest_loss = valid_loss / len(loader_valid)
            best_model = model
#
        early_stopping(lowest_loss, model)

        if early_stopping.early_stop:

            # calculate accuracy on test set
            #            all_y_valid, all_scores_valid, languages_valid, details_valid, all_tokens_valid = predict_one_by_one(best_model, valid_data)

            all_y_test, all_scores_test, languages_test, details_test, all_tokens_test = predict_one_by_one(
                best_model, test_data)

            run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + \
                '_RUN=' + str(number_run) + '_'

            test_ids = []
            with open(os.path.join(args.directory, 'test_documents.txt'), 'r') as f:
                for _id in f.readlines():
                    test_ids.append(_id.replace('\n', '').split('/')[-1])
            valid_ids = []
            with open(os.path.join(args.directory, 'valid_documents.txt'), 'r') as f:
                for _id in f.readlines():
                    valid_ids.append(_id.replace('\n', '').split('/')[-1])

            write_results_to_file(
                test_ids,
                run_name,
                epoch,
                all_tokens_test,
                details_test,
                all_y_test,
                all_scores_test,
                languages_test,
                'test',
                directory)
#            write_results_to_file(valid_ids, run_name, epoch, all_tokens_valid, details_valid, all_y_valid, all_scores_valid, 'valid', directory)
            break

    return np.average(p_test, weights=s_test) * 100.0, np.average(r_test, weights=s_test) * 100.0, \
        np.average(f1_test, weights=s_test) * 100.0, np.average(p_val, weights=s_val) * 100.0,\
        np.average(r_val, weights=s_val) * 100.0,\
        np.average(f1_val, weights=s_val) * 100.0


if __name__ == '__main__':

    #    pkl_dir = 'data/processed'

    files = [args.train, args.valid, args.test]
    assert os.path.exists(args.train)
    assert os.path.exists(args.test)
    assert os.path.exists(args.valid)
    assert os.path.exists(args.directory)
    assert os.path.exists(args.processed_directory)

    embeddings_type = args.embeddings
    model_name = args.model
    gold_test_file = args.gold_test
    multi_gpu = bool(args.multi_gpu)
#    write_results = bool(args.write_results)
    number_of_experiments = args.N
    max_sentence_len = args.max_len

    idx2wordPklPath = os.path.join(
        args.processed_directory,
        "idx2word_" + embeddings_type + ".pkl.gz")
    f = gzip.open(idx2wordPklPath, 'rb')
    idx2word = pkl.load(f)
    f.close()

    print(len(idx2word))

    word2IdxPklPath = os.path.join(
        args.processed_directory,
        "word2Idx_" + args.embeddings + ".pkl.gz")
    f = gzip.open(word2IdxPklPath, 'rb')
    word2Idx = pkl.load(f)
    f.close()

    idx2wordPklPath = os.path.join(
        args.processed_directory,
        "idx2word_" + args.embeddings + ".pkl.gz")
    f = gzip.open(idx2wordPklPath, 'rb')
    idx2word = pkl.load(f)
    f.close()

    char2idxPklPath = os.path.join(
        args.processed_directory,
        "char2idx_" + args.embeddings + ".pkl.gz")
    f = gzip.open(char2idxPklPath, 'rb')
    char2Idx = pkl.load(f)
    f.close()

    idx2charPklPath = os.path.join(
        args.processed_directory,
        "idx2char_" + args.embeddings + ".pkl.gz")
    f = gzip.open(idx2charPklPath, 'rb')
    idx2Char = pkl.load(f)
    f.close()

    generate_data = bool(args.generate_data)
    train_data, number_positions = prepare_dataset(args.train, max_sentence_len,
                                                   word2Idx,
                                                   char2Idx,
                                                   idx2Char,
                                                   augment=False,
                                                   type_data='train',
                                                   embeddings_type=embeddings_type,
                                                   generate_data=generate_data,
                                                   output_dir=args.processed_directory)

    number_positions = int(number_positions)
    logger.info('Maximum number of positions: {}'.format(number_positions))

    valid_data, _ = prepare_dataset(args.valid, max_sentence_len,
                                    word2Idx,
                                    char2Idx,
                                    idx2Char,
                                    augment=False,
                                    type_data='valid',
                                    embeddings_type=embeddings_type,
                                    generate_data=generate_data,
                                    output_dir=args.processed_directory)

    test_data, _ = prepare_dataset(args.test, max_sentence_len,
                                   word2Idx,
                                   char2Idx,
                                   idx2Char,
                                   augment=False,
                                   type_data='test',
                                   embeddings_type=embeddings_type,
                                   generate_data=generate_data,
                                   output_dir=args.processed_directory)

    torch.multiprocessing.set_sharing_strategy('file_system')
    dataset_train = TensorDataset(torch.from_numpy(np.array([x['token_matrix'] for x in train_data])),
                                  torch.from_numpy(np.array([x['char_matrix'] for x in train_data])),
                                  torch.from_numpy(np.array([x['position_matrix'] for x in train_data])),
                                  torch.from_numpy(np.array([x['labels'] for x in train_data])),
                                  torch.from_numpy(np.array([x['language'] for x in train_data])))

    loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
    )

    dataset_valid = TensorDataset(torch.from_numpy(np.array([x['token_matrix'] for x in valid_data])),
                                  torch.from_numpy(np.array([x['char_matrix'] for x in valid_data])),
                                  torch.from_numpy(np.array([x['position_matrix'] for x in valid_data])),
                                  torch.from_numpy(np.array([x['labels'] for x in valid_data])),
                                  torch.from_numpy(np.array([x['language'] for x in valid_data])))

    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size
    )

    dataset_test = TensorDataset(torch.from_numpy(np.array([x['token_matrix'] for x in test_data])),
                                 torch.from_numpy(np.array([x['char_matrix'] for x in test_data])),
                                 torch.from_numpy(np.array([x['position_matrix'] for x in test_data])),
                                 torch.from_numpy(np.array([x['labels'] for x in test_data])),
                                 torch.from_numpy(np.array([x['language'] for x in test_data])))

    loader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size
    )

    embeddings = None
    if not args.embeddings == 'none':
        f = gzip.open(os.path.join(args.processed_directory, 'embeddings_' +
                                   embeddings_type + '.pkl.gz'), 'rb')
        embeddings = pkl.load(f)
        embeddings = np.array(embeddings)
        f.close()
        print("Embeddings: ", embeddings.shape)
    else:
        size_embeddings = 300

    window_length = 31
    number_words = len(word2Idx)
    number_labels = len(INVERSE_MAPPING)
    print('number_labels', number_labels)
    size_embeddings = embeddings.shape[1]
    size_position_embeddings = 50

    precisions_test, recalls_test, f1s_test = [], [], []
    precisions_val, recalls_val, f1s_val = [], [], []
    for run in [1, 2, 3, 4, 5]:

        #        model = Attention_Net(size_embeddings, size_position_embeddings, window_length,
        # number_words, number_positions, number_labels, embeddings=embeddings)
        model = CNN_with_words(
            size_embeddings,
            size_position_embeddings,
            window_length,
            number_words,
            number_positions,
            number_labels,
            embeddings=embeddings)
#        model = CNN_with_chars(size_embeddings, window_length,
#                     len(char2Idx), number_labels, embeddings=embeddings)

        criterion = nn.CrossEntropyLoss()
        #criterion = nn.BCELoss()
        #criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        #optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.Adadelta(model.parameters())

        print('-' * 50, run, 'RUN')
        p, r, f1, p_val, r_val, f1_val = train_words(criterion, optimizer,
                                                     loader_train, loader_valid, loader_test,
                                                     valid_data, test_data,
                                                     number_run=str(run),
                                                     directory=os.path.join(args.processed_directory, 'results'))
        precisions_test.append(p)
        recalls_test.append(r)
        f1s_test.append(f1)
        precisions_val.append(p_val)
        recalls_val.append(r_val)
        f1s_val.append(f1_val)

    precisions_test, recalls_test, f1s_test = np.array(
        precisions_test), np.array(recalls_test), np.array(f1s_test)
    print("Precision: %0.2f (+/- %0.4f)" %
          (precisions_test.mean(), precisions_test.std() / 2))
    print("Recall: %0.2f (+/- %0.4f)" %
          (recalls_test.mean(), recalls_test.std() / 2))
    print("F1: %0.2f (+/- %0.4f)" % (f1s_test.mean(), f1s_test.std() / 2))

    precisions_val, recalls_val, f1s_val = np.array(
        precisions_val), np.array(recalls_val), np.array(f1s_val)
    print("Precision: %0.2f (+/- %0.4f)" %
          (precisions_val.mean(), precisions_val.std() / 2))
    print("Recall: %0.2f (+/- %0.4f)" %
          (recalls_val.mean(), recalls_val.std() / 2))
    print("F1: %0.2f (+/- %0.4f)" % (f1s_val.mean(), f1s_val.std() / 2))
