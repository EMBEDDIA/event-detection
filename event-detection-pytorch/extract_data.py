# -*- coding: utf-8 -*-

import MicroTokenizer
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, BertEmbeddings
import os
import embeddings as emb
import utils
from ast import literal_eval
from tqdm import tqdm
import argparse
from itertools import islice
import string as str_punctuation
from nltk.corpus import stopwords
import spacy
import pandas as pd
import gzip
from nltk import FreqDist
import pickle as pkl
import numpy as np
np.random.seed(42)  # for reproducibility

logger = utils.get_logger('Event Detection - data pre-process')


_inverseMapping = {0: 'Other', 1: 'End-Position', 2: 'Attack',
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

#_inverseMapping = {0: 'Other', 1: 'Event'}

labelsMapping = {v: k for k, v in _inverseMapping.items()}

language_mapping = {'en': 0, 'fr': 1, 'el': 2, 'pl': 3, 'zh': 4, 'ru': 5}

english_stopwords = set(stopwords.words('english'))

punctuation = set(str_punctuation.punctuation)

nlp = spacy.load('en_core_web_lg')

triggersDistribution = FreqDist()

distanceMapping = {}
minDistance = -15
maxDistance = 15

size_window = 31
half_window = 15

fixed_size = 150

distanceMapping['PADDING'] = 0
for dis in range(minDistance, maxDistance + 1):
    distanceMapping[dis] = len(distanceMapping)


def sliding_window(seq, n=5):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    slices = []
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        slices.append(result)
    for elem in it:
        result = result[1:] + (elem,)
        slices.append(result)

    return slices


def clean_text(text):
    doc = nlp(text)
    # if x.text not in english_stopwords and (x.text not in punctuation)])
    tokens = list([x for x in doc])
    return tokens


def _get_stopwords(sentence):
    return list(set([x for x in nlp(sentence)
                     if x.text.lower() in punctuation]))


def get_entities(sentence):
    doc = nlp(sentence)
    entities = []
    for ent in doc.ents:
        for split in ent.text.split(' '):
            entities.append(ent.text)

    return entities


def read_lexicon(filename):
    lexicon = {}
    all_words = []
    for line in open(filename, 'r'):
        words = line.strip().split()
        if words[0] not in ['Other']:
            if words[0] not in lexicon:
                lexicon[words[0]] = []
            for word in words[1:]:
                all_words.append(word.lower())
                lexicon[words[0]].append(word.lower())

    return all_words, lexicon


def prepare_dataset(file,
                    maxSentenceLen,
                    word2Idx,
                    char2Idx,
                    idx2Char,
                    augment=False,
                    type_data='train',
                    embeddings_type='google',
                    generate_data=False,
                    output_dir=None):
    print(file)
    """Creates matrices for the events and sentence for the given file"""
    data_pkl_path = os.path.join(
        output_dir,
        "data_" +
        type_data +
        '_' +
        embeddings_type +
        ".pkl.gz")
    if generate_data:

        if augment:
            all_words, lexicon = read_lexicon('data/augment.txt')

        labelsDistribution = FreqDist()
        triggersDistribution = FreqDist()

        triggers_here = {}
        logger.info('Processing:' + file)

        labels, labels_sentence, positionMatrix, tokenMatrix, charMatrix = [], [], [], [], []
        whole_sentence_matrix, all_sentence_matrix, details_matrix, triggers_out = [], [], [], []

        check_existing = []
        df = pd.read_csv(
            file,
            sep='\t',
            names=[
                'doc_id',
                'type',
                'subtype',
                'old_index',
                'sent',
                'anchor_text',
                'anchors',
                'language'],
            converters={
                "anchors": literal_eval})  # .sample(frac=.1) #TODO

        no_tokens = 0
        no_sentences = 0
        triggers = {}

        data = []

        logger.info(str(len(pd.unique(df.sent))) + ' sentences')

        for index, line in tqdm(df.iterrows(), total=len(df)):
            sentence = line.sent
            doc_id = line.doc_id
            language = 'en'  # line.language

            anchors = []
            anchors_indexes = {}
            event_ids = {}
            anchors_labels = dict()

            for df_anchors in np.unique(df[df.sent == sentence].anchors):
                if len(df_anchors) > 0:
                    anchor, label, positions, event_subtype = df_anchors[0]
                    start, end = positions.split(',')

                    x1 = anchor.split(' ')
                    x2 = anchor.split('-')

                    if isinstance(x1, list):
                        for x in x1:
                            if x not in sentence:
                                x = x.lower()
                            if x in anchors:
                                anchors_indexes[x] = [anchors_indexes[x]]
                                event_ids[x] = [event_ids[x]]
                                anchors_indexes[x].append(start + ',' + end)
                                event_ids[x].append(event_subtype)
                            else:
                                anchors.append(x)
                                anchors_labels[x] = label
                                event_ids[x] = event_subtype
                                anchors_indexes[x] = start + ',' + end
                    elif isinstance(x2, list):
                        for x in x2:
                            if x not in sentence:
                                x = x.lower()
                            if x in anchors:
                                anchors_indexes[x] = [anchors_indexes[x]]
                                event_ids[x] = [event_ids[x]]
                                anchors_indexes[x].append(start + ',' + end)
                                event_ids[x].append(event_subtype)
                            else:
                                anchors.append(x)
                                anchors_labels[x] = label
                                event_ids[x] = event_subtype
                                anchors_indexes[x] = start + ',' + end
                    else:
                        x = anchor
                        if anchor not in sentence:
                            x = anchor.lower()
                        if x in anchors:
                            anchors_indexes[x] = [anchors_indexes[x]]
                            event_ids[x] = [event_ids[x]]
                            anchors_indexes[x].append(start + ',' + end)
                            event_ids[x].append(event_subtype)
                        else:
                            anchors.append(x)
                            anchors_labels[x] = label
                            event_ids[x] = event_subtype
                            anchors_indexes[x] = start + ',' + end

            label = line.subtype
            if line.subtype == 'None':
                label = 'Other'

            # don't take the same sentence w/ the same trigger
            if sentence not in check_existing:
                no_sentences += 1
                take_sentence = True
                if len(anchors) > 0:
                    check_existing.append(sentence)
                    print(anchors)

                doc = nlp(sentence)

                if language == 'zh':
                    clean_tokens = MicroTokenizer.cut(sentence, HMM=True)

                clean_tokens = [x.text for x in clean_text(sentence)]

                no_tokens += len(clean_tokens)

    #            stopwords = [x.text for x in  _get_stopwords(sentence)]

                pos = {}
                tag = {}
                for x in doc:
                    pos[x.text] = x.pos_
                    tag[x.text] = x.tag_

    #            entities = get_entities(sentence)

                allTokenIds = []

                previous_token_ids = np.zeros(fixed_size)

                for idx in range(0, len(clean_tokens)):
                    allTokenIds.append(getWordIdx(clean_tokens[idx], word2Idx))

                if len(clean_tokens) < fixed_size:
                    for idx in range(0, len(clean_tokens)):
                        previous_token_ids[idx] = getWordIdx(
                            clean_tokens[idx], word2Idx)
                    for idx in range(len(clean_tokens), fixed_size):
                        previous_token_ids[idx] = getWordIdx(
                            'PADDING', word2Idx)
                else:
                    for idx in range(0, fixed_size):
                        previous_token_ids[idx] = getWordIdx(
                            clean_tokens[idx], word2Idx)

                for i in range(half_window):
                    clean_tokens.append("PADDING")
                    clean_tokens.insert(0, "PADDING")

                    allTokenIds.append(getWordIdx("PADDING", word2Idx))
                    allTokenIds.insert(0, getWordIdx("PADDING", word2Idx))

                all_sentence_matrix.append(np.array(previous_token_ids))

                if len(anchors) > 0:
                    labels_sentence.append(1)
                else:
                    labels_sentence.append(0)

                assert len(clean_tokens) == len(allTokenIds)
                if line.subtype not in triggers_here:
                    triggers_here[line.subtype] = FreqDist()

                if take_sentence:
                    found_anchors = []
                    for value in zip(sliding_window(clean_tokens, size_window),
                                     sliding_window(allTokenIds, size_window)):

                        details_event = np.zeros(3, dtype=object)
                        details_event[0] = str(doc_id)

                        window, tokenIds = value
                        window = list(window)
                        tokenIds = list(tokenIds)

                        target_word = window[half_window]

                        if target_word == 'PADDING':
                            continue

                        take = True
                        how_many_times = 1

                        if target_word in anchors:

                            #                            if "Le diagnostic de l'OMS contredit les prop" in sentence:
                            #                                import pdb;pdb.set_trace()
                            if target_word in anchors_indexes:
                                if isinstance(
                                        anchors_indexes[target_word], list):
                                    if len(event_ids[target_word]) > 0:
                                        label = anchors_labels[target_word]
                                        details_event[1] = str(
                                            event_ids[target_word][0])
                                        details_event[2] = str(
                                            anchors_indexes[target_word][0])

                                        del event_ids[target_word][0]
                                        del anchors_indexes[target_word][0]
                                else:
                                    label = anchors_labels[target_word]
                                    found_anchors.append(target_word)

                                    details_event[1] = str(
                                        event_ids[target_word])
                                    details_event[2] = str(
                                        anchors_indexes[target_word])

                                    del anchors_indexes[target_word]
                                    del event_ids[target_word]
                            else:
                                label = 'Other'
                                details_event[1] = 'Other'
                                details_event[2] = str(sentence.index(
                                    target_word)) + ',' + str(sentence.index(target_word) + len(target_word))
                        else:
                            label = 'Other'
                            details_event[1] = 'Other'
                            details_event[2] = str(sentence.index(
                                target_word)) + ',' + str(sentence.index(target_word) + len(target_word))

                        windows = [(label, tokenIds)]
                        if label not in ['Other']:
                            print(target_word, '--', language)
#                        print(details_event[2])
#                        if '60,68' in str(details_event[2]):
#                            print(details_event)
#                            import pdb;pdb.set_trace()

                        if augment:
                            ###################################################
                            # DATA AUG
                            if 'train' in file:
                                if label not in ['Other']:
                                    if target_word.lower() in lexicon:
                                        for word in lexicon[target_word.lower(
                                        )]:
                                            tokenIds2 = []
                                            for idx_, tokenId in enumerate(
                                                    tokenIds):
                                                if idx_ == half_window:
                                                    tokenIds2.append(
                                                        getWordIdx(word, word2Idx))
                                                else:
                                                    tokenIds2.append(tokenId)
                                            windows.append((label, tokenIds2))
                                    else:
                                        if target_word.lower() in all_words:
                                            for key, value in lexicon.items():
                                                if target_word.lower() in value:
                                                    for word in value:
                                                        tokenIds2 = []
                                                        for idx_, tokenId in enumerate(
                                                                tokenIds):
                                                            if idx_ == half_window:
                                                                tokenIds2.append(
                                                                    getWordIdx(word, word2Idx))
                                                            else:
                                                                tokenIds2.append(
                                                                    tokenId)
                                                        windows.append(
                                                            (label, tokenIds2))
        #                                                print('Added 2:', word, label)
                              #################################################

                        if take:
                            for window_ in windows:
                                label_ = window_[0]
                                tokenIds = window_[1]

                                for how in range(how_many_times):
                                    if label_ not in ['Other']:
                                        triggersDistribution[target_word] += 1

                                    positionValues = np.zeros(size_window)

                                    distances = list(
                                        range(-half_window, half_window + 1))

                                    size_char = 256
                                    all_char = np.zeros(size_char)
                                    i = 0
#                                    what = ''

                                    for idx in range(len(window)):
                                        #                                        what += ' '
                                        #                                        if len(window[idx]) > 7:
                                        #                                            word = window[idx][:7]
                                        #                                        if len(window[idx]) < 7:
                                        #                                            word = window[idx] + (7-len(window[idx]))*' '
                                        if i == size_char:
                                            break

                                        for char in window[idx]:
                                            if i == size_char:
                                                break

                                            if clean_tokens[idx] in [
                                                    'PADDING', 'UNKNOWN']:
                                                all_char[i] = getCharIdx(
                                                    0, char2Idx)
#                                                what += ' '
                                            else:
                                                if window[idx] not in [
                                                        'PADDING', 'UNKNOWN']:
                                                    try:
                                                        all_char[i] = getCharIdx(
                                                            char, char2Idx)
#                                                        what += char
                                                    except BaseException:
                                                        import pdb
                                                        pdb.set_trace()

                                            i += 1

                                    labelsDistribution[label_] += 1

                                    if label_ not in triggers:
                                        triggers[label_] = FreqDist()
                                    triggers[label_][target_word] += 1

                                    lexicalIds = np.zeros(3)
                                    lexicalIds[0] = tokenIds[half_window - 1]
                                    lexicalIds[1] = tokenIds[half_window]
                                    lexicalIds[2] = tokenIds[half_window + 1]

                                    for idx in range(size_window):
                                        if window[idx] == 'PADDING':
                                            positionValues[idx] = distanceMapping['PADDING']
                                        else:
                                            positionValues[idx] = distanceMapping[distances[idx]]

                                    data.append({
                                        'details_event': details_event,
                                        'char_matrix': np.asarray(all_char),
                                        'token_matrix': tokenIds,
                                        'position_matrix': positionValues,
                                        'labels': labelsMapping[label_],
                                        'language': language_mapping[language],
                                    })

                                    positionMatrix.append(positionValues)

        print('Saving processed data in', output_dir)

        max_position = np.max(positionMatrix) + 1

        f = gzip.open(data_pkl_path, 'wb')
        pkl.dump(data, f, -1)
        pkl.dump(max_position, f, -1)
        f.close()

    else:
        f = gzip.open(data_pkl_path, 'rb')
        data = pkl.load(f)
        max_position = pkl.load(f)
        f.close()

    return data, max_position


def getCharIdx(char, char2Idx):
    if char in char2Idx:
        return char2Idx[char]
    else:
        return char2Idx['PADDING']


def getPosIdx(pos, pos2Idx):
    if pos in pos2Idx:
        return pos2Idx[pos]
    else:
        return pos2Idx['UNKNOWN']


def getWordIdx(token, word2Idx):
    """Returns from the word2Idex table the word index for a given token"""
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]

    return word2Idx["UNKNOWN"]


def add_unknown_words(
        embeddings,
        words,
        word2Idx,
        idx2Word,
        vocab,
        min_df=5,
        k=300,
        mean=0,
        std=0):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """

    ##embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    added_words = 0
    for word in vocab:
        if word not in word2Idx:  # and vocab[word] >= min_df:
            #            vector = np.random.normal(loc=mean, scale=std, size=k)
            vector = np.random.normal(loc=mean, scale=std, size=k)
            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)
            idx2Word[word2Idx[word]] = word
            words[word] = True
            added_words += 1

    return added_words, len(word2Idx)


def create_dictionary(files, embeddings_model, output_dir):
    embeddings, word2Idx, pos2Idx, idx2Pos, words, idx2word, char2Idx, idx2Char = \
        [], {}, {}, {}, {}, {}, {}, {}

    maxSentenceLen = 0
    vocab_in_text = {}
    for fileIdx in range(len(files)):
        already_parsed = []
        file = files[fileIdx]
        logger.info('Pre-processing {}'.format(file))

        df = pd.read_csv(
            file,
            sep='\t',
            names=[
                'doc_id',
                'type',
                'subtype',
                'blabla',
                'sent',
                'anchor_text',
                'anchors',
                'language'])

        for idx, line in tqdm(df.iterrows(), total=len(df)):
            sentence = line.sent
#            doc = nlp(sentence)
            language = line.language

            if sentence not in already_parsed:
                already_parsed.append(sentence)

                if len(char2Idx) == 0:  # Add padding+unknown
                    char2Idx["PADDING"] = len(char2Idx)
                    char2Idx["PUNCT"] = len(char2Idx)
                    char2Idx["UNKNOWN"] = len(char2Idx)

                if len(pos2Idx) == 0:  # Add padding+unknown
                    pos2Idx["PADDING"] = len(pos2Idx)
                    idx2Pos[pos2Idx["PADDING"]] = 'PADDING'

                if len(word2Idx) == 0:  # Add padding+unknown

                    if 'bert' not in args.embeddings:
                        # Zero vector vor 'PADDING' word
                        vector = np.zeros(size_embeddings)
                        if 'PADDING' in embeddings_model:
                            embeddings.append(embeddings_model['PADDING'])
                        else:
                            embeddings.append(vector)
                        word2Idx["PADDING"] = len(word2Idx)
                        idx2word[word2Idx["PADDING"]] = "PADDING"

                        #vector = np.random.uniform(-max_embed, max_embed, size_embeddings)
                        vector = np.random.normal(
                            loc=emb_mean, scale=emb_std, size=size_embeddings)
                        if 'UNKNOWN' in embeddings_model:
                            embeddings.append(embeddings_model['UNKNOWN'])
                        else:
                            embeddings.append(vector)
                        word2Idx["UNKNOWN"] = len(word2Idx)
                        idx2word[word2Idx["UNKNOWN"]] = "UNKNOWN"

                    else:
                        vector = np.zeros(size_embeddings)
                        embeddings.append(vector)
                        word2Idx["PADDING"] = len(word2Idx)
                        idx2word[word2Idx["PADDING"]] = "PADDING"

                        vector = np.random.uniform(-np.sqrt(0.06),
                                                   np.sqrt(0.06), size_embeddings)
                        embeddings.append(vector)
                        word2Idx["UNKNOWN"] = len(word2Idx)
                        idx2word[word2Idx["UNKNOWN"]] = "UNKNOWN"

                    # Add also the words from augmentation
                    # from the pre-trained embeddings
                    if args.augment:
                        all_words, lexicon = read_lexicon('data/augment.txt')

                        for token in all_words:
                            if token in vocab:
                                if token not in word2Idx:
                                    embeddings.append(embeddings_model[token])
                                    word2Idx[token] = len(word2Idx)
                                    idx2word[word2Idx[token]] = token
                                    words[token] = True
                doc = nlp(sentence)
                tokens = [x.text for x in doc]
                if len(tokens) > maxSentenceLen:
                    maxSentenceLen = len(tokens)

                if language == 'zh':
                    tokens = MicroTokenizer.cut(sentence, HMM=True)

                if 'bert' in args.embeddings:
                    try:
                        sentence = Sentence(' '.join([x for x in tokens]))
                        bert_embedding.embed(sentence)  # 3072
                        for token in sentence:
                            if token not in word2Idx:
                                embeddings.append(
                                    token.embedding.cpu().numpy())
                                word2Idx[token] = len(word2Idx)
                                idx2word[word2Idx[token]] = token
                                words[token] = True
                                if token in vocab_in_text:
                                    vocab_in_text[token] += 1
                                else:
                                    vocab_in_text[token] = 1
                                for char in token:
                                    if char not in char2Idx:
                                        char2Idx[char] = len(char2Idx)
                                        idx2Char[char2Idx[char]] = char
                    except BaseException:
                        print(' '.join([x for x in tokens]))
                else:
                    for idx, word in enumerate(tokens):
                        token = word
#                        lemma = word.lemma_
    #
                        for char in token:
                            if char not in char2Idx:
                                char2Idx[char] = len(char2Idx)
                                idx2Char[char2Idx[char]] = char

                        if token in vocab:
                            if token not in word2Idx:
                                embeddings.append(embeddings_model[token])
                                word2Idx[token] = len(word2Idx)
                                idx2word[word2Idx[token]] = token
                                words[token] = True

                        else:
                            if token not in word2Idx:
                                if token.lower() in vocab:
                                    embeddings.append(
                                        embeddings_model[token.lower()])
                                    word2Idx[token] = len(word2Idx)
                                    idx2word[word2Idx[token]] = token
                                    words[token] = True
#                                elif lemma in vocab:
#                                    embeddings.append(embeddings_model[lemma])
#                                    word2Idx[token] = len(word2Idx)
#                                    idx2word[word2Idx[token]] = token
#                                    words[token] = True
                                elif token[0].upper() + token[1:] in vocab:
                                    embeddings.append(
                                        embeddings_model[token[0].upper() + token[1:]])
                                    word2Idx[token] = len(word2Idx)
                                    idx2word[word2Idx[token]] = token
                                    words[token] = True

                        if token in vocab_in_text:
                            vocab_in_text[token] += 1
                        else:
                            vocab_in_text[token] = 1

    unknown_words = 0
    for word in vocab_in_text.keys():
        if word not in word2Idx:
            unknown_words += 1

    print(
        'Generating embeddings for unknown words. Known words: {} Unknown words: {}'.format(
            len(word2Idx),
            unknown_words))
    if 'bert' not in args.embeddings:
        x, y = add_unknown_words(embeddings, words, word2Idx, idx2word, vocab_in_text,
                                 k=size_embeddings, mean=emb_mean, std=emb_std)
        print(x, y)

    print(len(word2Idx))
    print('Saving processed data in', output_dir)
    word2IdxPklPath = os.path.join(
        output_dir,
        "word2Idx_" +
        args.embeddings +
        ".pkl.gz")
    f = gzip.open(word2IdxPklPath, 'wb')
    pkl.dump(word2Idx, f, -1)
    f.close()

    idx2wordPklPath = os.path.join(
        output_dir,
        "idx2word_" +
        args.embeddings +
        ".pkl.gz")
    f = gzip.open(idx2wordPklPath, 'wb')
    pkl.dump(idx2word, f, -1)
    f.close()

    char2idxPklPath = os.path.join(
        output_dir,
        "char2idx_" +
        args.embeddings +
        ".pkl.gz")
    f = gzip.open(char2idxPklPath, 'wb')
    pkl.dump(char2Idx, f, -1)
    f.close()

    idx2charPklPath = os.path.join(
        output_dir,
        "idx2char_" +
        args.embeddings +
        ".pkl.gz")
    f = gzip.open(idx2charPklPath, 'wb')
    pkl.dump(idx2Char, f, -1)
    f.close()

    f = gzip.open(embeddingsPklPath, 'wb')
    pkl.dump(embeddings, f, -1)
    f.close()

    vocab_metadata = 'data/vocab.txt'
    f_vocab = open(vocab_metadata, 'w')
    for i, j in idx2word.items():
        f_vocab.write(j + '\n')
    f_vocab.close()

    f = gzip.open(embeddingsPklPath, 'rb')
    embeddings = pkl.load(f)
    embeddings = np.array(embeddings)
    f.close()

    logger.info('Maximum sentence length {}'.format(maxSentenceLen))
    logger.info('Total words {}'.format(len(word2Idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trigger pre-processing: Create dictionary and store data in binary format')
    parser.add_argument(
        '--embeddings',
        default='google',
        help='Embeddings type: google, glove, fasttext and numberbatch')
    parser.add_argument('--train', default='data/train.txt',
                        help='Train file')
    parser.add_argument('--test', default='data/test.txt',
                        help='Test file')
    parser.add_argument('--valid', default='data/valid.txt',
                        help='Validation file')
    parser.add_argument('--augment', default=False,
                        help='Augment or not')
    parser.add_argument('--output_directory', default='data',
                        help='Augment or not')

    args = parser.parse_args()

    files = [args.train, args.valid, args.test]
    assert os.path.exists(args.train)
    assert os.path.exists(args.test)
    assert os.path.exists(args.valid)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    logger.info('Loading pretrained embeddings {}'.format(args.embeddings))

    embeddingsPklPath = os.path.join(
        args.output_directory,
        'embeddings_' +
        args.embeddings +
        '.pkl.gz')

    if 'glove' in args.embeddings:
        embeddings_model = emb.fetch_GloVe()
    elif 'google' in args.embeddings:
        embeddings_model = emb.fetch_SG_GoogleNews()
    elif 'fast' in args.embeddings:
        embeddings_model = emb.fetch_FastText()
    elif 'number' in args.embeddings:
        embeddings_model = emb.fetch_conceptnet_numberbatch()
    elif 'bert' in args.embeddings:
        size_embeddings = 3072

    if 'bert' not in args.embeddings:
        emb_std = np.std(embeddings_model.vectors)
        emb_mean = np.mean(embeddings_model.vectors)

    if 'bert' not in args.embeddings:
        try:
            vocab = embeddings_model.vocabulary
        except BaseException:
            vocab = embeddings_model.vocab

        try:
            size_embeddings = embeddings_model.shape[1]
        except BaseException:
            size_embeddings = embeddings_model.vector_size

    else:
        bert_embedding = BertEmbeddings('bert-base-multilingual-cased')
        embeddings_model = None

    create_dictionary(files, embeddings_model, args.output_directory)


#    word2IdxPklPath = "data/processed/word2Idx_"+args.embeddings+".pkl.gz"
#    f = gzip.open(word2IdxPklPath, 'rb')
#    word2Idx = pkl.load(f)
#    f.close()
#
#    idx2wordPklPath = "data/processed/idx2word_"+args.embeddings+".pkl.gz"
#    f = gzip.open(idx2wordPklPath, 'rb')
#    idx2word = pkl.load(f)
#    f.close()
#
#    char2idxPklPath = "data/processed/char2idx_"+args.embeddings+".pkl.gz"
#    f = gzip.open(char2idxPklPath, 'rb')
#    char2Idx = pkl.load(f)
#    f.close()
#
#    idx2charPklPath = "data/processed/idx2char_"+args.embeddings+".pkl.gz"
#    f = gzip.open(idx2charPklPath, 'rb')
#    idx2Char = pkl.load(f)
#    f.close()
#
#    maxSentenceLen = 57
#
#    train_set = create_matrices(files[0], word2Idx, maxSentenceLen,
#                                char2Idx, idx2Char, args.augment)
#    f = gzip.open('data/processed/triggers_' +
#                  args.embeddings + '_train.pkl.gz', 'wb')
#    pkl.dump(train_set, f, -1)
#    f.close()
#
#    valid_set = create_matrices(files[1], word2Idx, maxSentenceLen,
#                                char2Idx, idx2Char, args.augment)
#    f = gzip.open('data/processed/triggers_' +
#                  args.embeddings + '_valid.pkl.gz', 'wb')
#    pkl.dump(valid_set, f, -1)
#    f.close()
#
#    test_set = create_matrices(files[2], word2Idx, maxSentenceLen,
#                               char2Idx, idx2Char, args.augment)
#    f = gzip.open('data/processed/triggers_' +
#                  args.embeddings + '_test.pkl.gz', 'wb')
#    pkl.dump(test_set, f, -1)
#    f.close()
# print(idx2Char)
#    #print('VOCAB', len(vocab))
