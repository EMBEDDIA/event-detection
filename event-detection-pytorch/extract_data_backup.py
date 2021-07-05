# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(42)  # for reproducibility
import pickle as pkl
from nltk import FreqDist
import gzip
import pandas as pd
import spacy
from nltk.corpus import stopwords
import string as str_punctuation
from itertools import islice
import argparse
from tqdm import tqdm
from ast import literal_eval
import utils
import embeddings as emb
import os
logger = utils.get_logger('Event Detection - data pre-process')
from flair.embeddings import FlairEmbeddings, BertEmbeddings

bert_embedding = BertEmbeddings('bert-base-multilingual-cased')
from flair.data import Sentence

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

labelsMapping = {v: k for k, v in _inverseMapping.items()}

english_stopwords = set(stopwords.words('english'))

punctuation = set(str_punctuation.punctuation)

nlp = spacy.load('en')

maxSentenceLen = [0, 0, 0]
triggersDistribution = FreqDist()

distanceMapping = {}
minDistance = -15
maxDistance = 15

size_window = 31
half_window = 15

fixed_size = 150

distanceMapping['PADDING'] = 0
for dis in range(minDistance, maxDistance+1):
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
    tokens = list([x for x in doc if x.text not in [' ']])
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


def create_matrices(file, word2Idx,
                    maxSentenceLen,
                    char2Idx,
                    idx2Char,
                    augment=False,
                    embeddings_type=None):
    """Creates matrices for the events and sentence for the given file"""

    if augment:
        all_words, lexicon = read_lexicon('data/augment.txt')

    labelsDistribution = FreqDist()
    triggersDistribution = FreqDist()

    triggers_here = {}
    logger.info('Processing:' + file)

    labels, labels_sentence, positionMatrix, tokenMatrix, charMatrix = [], [], [], [], []
    whole_sentence_matrix, all_sentence_matrix, details_matrix, triggers_out = [], [], [], []

    check_existing = []
    df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'old_index',
                                            'sent', 'anchor_text', 'anchors'],
                     converters={"anchors": literal_eval})

    no_tokens = 0
    no_sentences = 0
    triggers = {}

    logger.info(str(len(pd.unique(df.sent))) + ' sentences')

    for index, line in tqdm(df.iterrows(), total=len(df)):
        sentence = line.sent
        doc_id = line.doc_id

        anchors = []
        anchors_indexes = {}
        event_ids = {}
        anchors_labels = dict()

        for df_anchors in df[df.sent == sentence].anchors:
            if len(df_anchors) > 0:
                anchor, label, positions, event_subtype = df_anchors[0]
                start, end = positions.split(',')

                x1 = anchor.split(' ')
                x2 = anchor.split('-')

                if type(x1) == list:
                    for x in x1:
                        if x not in sentence:
                            x = x.lower()
                        if x in anchors:
                            anchors_indexes[x] = [anchors_indexes[x]]
                            event_ids[x] = [event_ids[x]]
                            anchors_indexes[x].append(start+','+end)
                            event_ids[x].append(event_subtype)
                        else:
                            anchors.append(x)
                            anchors_labels[x] = label
                            event_ids[x] = event_subtype
                            anchors_indexes[x] = start+','+end
                elif type(x2) == list:
                    for x in x2:
                        if x not in sentence:
                            x = x.lower()
                        if x in anchors:
                            anchors_indexes[x] = [anchors_indexes[x]]
                            event_ids[x] = [event_ids[x]]
                            anchors_indexes[x].append(start+','+end)
                            event_ids[x].append(event_subtype)
                        else:
                            anchors.append(x)
                            anchors_labels[x] = label
                            event_ids[x] = event_subtype
                            anchors_indexes[x] = start+','+end
                else:
                    x = anchor
                    if anchor not in sentence:
                        x = anchor.lower()
                    if x in anchors:
                        anchors_indexes[x] = [anchors_indexes[x]]
                        event_ids[x] = [event_ids[x]]
                        anchors_indexes[x].append(start+','+end)
                        event_ids[x].append(event_subtype)
                    else:
                        anchors.append(x)
                        anchors_labels[x] = label
                        event_ids[x] = event_subtype
                        anchors_indexes[x] = start+','+end

        label = line.subtype
        if line.subtype == 'None':
            label = 'Other'

        # don't take the same sentence w/ the same trigger
        if sentence not in check_existing:
            no_sentences += 1
            take_sentence = True
            if len(anchors) > 0:
                check_existing.append(sentence)

            doc = nlp(sentence)
            clean_tokens = [x.text for x in clean_text(sentence)]
            no_tokens += len(clean_tokens)
            
            if ' ' in clean_tokens:
                import pdb;pdb.set_trace()

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
                allTokenIds.append(getWordIdx(clean_tokens[idx], word2Idx, embeddings_type, clean_tokens))

            if len(clean_tokens) < fixed_size:
                for idx in range(0, len(clean_tokens)):
                    previous_token_ids[idx] = getWordIdx(
                        clean_tokens[idx], word2Idx, embeddings_type, clean_tokens)
                for idx in range(len(clean_tokens), fixed_size):
                    previous_token_ids[idx] = getWordIdx('PADDING', word2Idx, embeddings_type, clean_tokens)
            else:
                for idx in range(0, fixed_size):
                    previous_token_ids[idx] = getWordIdx(
                        clean_tokens[idx], word2Idx, embeddings_type, clean_tokens)

            for i in range(half_window):
                clean_tokens.append("PADDING")
                clean_tokens.insert(0, "PADDING")

                allTokenIds.append(getWordIdx("PADDING", word2Idx, embeddings_type, clean_tokens))
                allTokenIds.insert(0, getWordIdx("PADDING", word2Idx, embeddings_type, clean_tokens))

            all_sentence_matrix.append(np.array(previous_token_ids))

            if len(anchors) > 0:
                labels_sentence.append(1)
            else:
                labels_sentence.append(0)

            assert len(clean_tokens) == len(allTokenIds)
            if line.subtype not in triggers_here:
                triggers_here[line.subtype] = FreqDist()

            if take_sentence == True:
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
                        if target_word in anchors_indexes:
                            if type(anchors_indexes[target_word]) == list:
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

                                details_event[1] = str(event_ids[target_word])
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

                    if augment:
                        #######################################################
                        # DATA AUG
                        if 'train' in file:
                            if label not in ['Other']:
                                if target_word.lower() in lexicon:
                                    for word in lexicon[target_word.lower()]:
                                        tokenIds2 = []
                                        for idx_, tokenId in enumerate(tokenIds):
                                            if idx_ == half_window:
                                                tokenIds2.append(
                                                    getWordIdx(word, word2Idx, embeddings_type, clean_tokens))
                                            else:
                                                tokenIds2.append(tokenId)
                                        windows.append((label, tokenIds2))
                                else:
                                    if target_word.lower() in all_words:
                                        for key, value in lexicon.items():
                                            if target_word.lower() in value:
                                                for word in value:
                                                    tokenIds2 = []
                                                    for idx_, tokenId in enumerate(tokenIds):
                                                        if idx_ == half_window:
                                                            tokenIds2.append(
                                                                getWordIdx(word, word2Idx, embeddings_type, clean_tokens))
                                                        else:
                                                            tokenIds2.append(
                                                                tokenId)
                                                    windows.append(
                                                        (label, tokenIds2))
    #                                                print('Added 2:', word, label)
                          #####################################################

                    if take == True:
                        for window_ in windows:
                            label_ = window_[0]
                            tokenIds = window_[1]

                            for how in range(how_many_times):
                                if label_ not in ['Other']:
                                    triggersDistribution[target_word] += 1

                                positionValues = np.zeros(size_window)

                                distances = list(
                                    range(-half_window, half_window+1))

                                all_char = np.zeros(1024)
                                i = 0
                                what = ''
                                
#                                for idx in range(size_window):
#                                    charValues = np.zeros(7)
#                                    for i in range(7):
#                                        charValues[i] = getCharIdx(0, char2Idx)

                                for idx in range(len(window)):
                                    what += ' '
                                    if len(window[idx]) > 7:
                                        word = window[idx][:7]
                                    if len(window[idx]) < 7:
                                        word = window[idx] + (7-len(window[idx]))*' '
                                        
                                    for char in window[idx]:
                                        
                                        if clean_tokens[idx] in ['PADDING', 'UNKNOWN']:
                                            all_char[i] = getCharIdx(0, char2Idx)
                                            what += ' '
                                        else:
                                            if window[idx] not in ['PADDING', 'UNKNOWN']:
                                                all_char[i] = getCharIdx(
                                                    char, char2Idx)
                                                what += char
                                        i += 1
                                        
#                                print(' '.join([x for x in clean_tokens if x not in ['PADDING', 'UNKNOWN']]))
#                                print(target_word + ':' + ' '.join([x for x in window if x not in ['PADDING', 'UNKNOWN']]))
#                                print(what)
#                                print()
#                                import pdb;pdb.set_trace()
                                labelsDistribution[label_] += 1

                                if label_ not in triggers:
                                    triggers[label_] = FreqDist()
                                triggers[label_][target_word] += 1

                                lexicalIds = np.zeros(3)
                                lexicalIds[0] = tokenIds[half_window-1]
                                lexicalIds[1] = tokenIds[half_window]
                                lexicalIds[2] = tokenIds[half_window+1]

                                for idx in range(size_window):
                                    if window[idx] == 'PADDING':
                                        positionValues[idx] = distanceMapping['PADDING']
                                    else:
                                        positionValues[idx] = distanceMapping[distances[idx]]

                                triggers_out.append(target_word)
                                details_matrix.append(details_event)
                                charMatrix.append(np.asarray(all_char))
                                tokenMatrix.append(tokenIds)
                                positionMatrix.append(positionValues)
                                labels.append(labelsMapping[label_])
                                whole_sentence_matrix.append(
                                    np.array(previous_token_ids))

    print(file, 'sentences:', no_sentences, 'tokens:', no_tokens)
    print("Data stored in pkl folder", len(labelsDistribution))

#    for label, freq in triggersDistribution.most_common(100):
#        print( "%s : %f%%" % (label, 50*freq / float(labelsDistribution.N())))
#    triggersDistribution.plot(30, cumulative=False)

    for label, freq in labelsDistribution.most_common(100):
        print("%s : %f%%" % (label, 100*freq / float(labelsDistribution.N())))
#    labelsDistribution.plot(30, cumulative=False)

#    with open(file.replace('.txt', '_triggers.txt'), 'w') as f_file:
#        for idx, value in triggers.items():
#            if 'Other' not in idx:
#                f_file.write(str(idx) + ' ')
#                for label, freq in value.most_common():
#                    if 'Other' not in label:
#                        f_file.write(str(label) + ' ')
#                f_file.write('\n')

    return _inverseMapping, \
        np.array(details_matrix, dtype=object), \
        np.array(labels, dtype='int32'), \
        np.array(tokenMatrix, dtype='int32'), \
        np.array(positionMatrix, dtype='int32'), \
        np.array(charMatrix, dtype='int32'), \
        np.array(whole_sentence_matrix, dtype='int32'), \
        np.array(all_sentence_matrix, dtype='int32'), \
        np.array(labels_sentence, dtype='int32'), \
        np.array(triggers_out)


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


def getWordIdx(token, word2Idx, embeddings_type, sentence_tokens):
    """Returns from the word2Idex table the word index for a given token"""
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    if 'bert' in embeddings_type:
        print(token)
#        import pdb;pdb.set_trace()
        sentence_ = Sentence(token)
        bert_embedding.embed(sentence_)#3072
        if len(sentence_.tokens) == 0:
            word2Idx["UNKNOWN"]
        for token_ in sentence_:
            if token_.text not in word2Idx:
#                embeddings = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
                embeddings.append(token_.embedding.cpu().numpy())
                word2Idx[token_.text] = len(word2Idx)
                idx2word[word2Idx[token_.text]] = token_.text
                words[token_.text] = True
                return word2Idx[token_.text]

    return word2Idx["UNKNOWN"]


def add_unknown_words(embeddings, words, word2Idx, idx2Word, vocab, min_df=5, k=300, mean=0, std=0):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """

    ##embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word in vocab:
        if word not in word2Idx:  # and vocab[word] >= min_df:
#            vector = np.random.normal(loc=mean, scale=std, size=k)
            vector = np.random.normal(loc=mean, scale=std, size=k)
            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)
            idx2Word[word2Idx[word]] = word
            words[word] = True


#labelsPath = 'data/labels.txt'
#eventTypesPath = 'data/event_types.txt'
#eventSubtypesPath = 'data/event_subtypes.txt'
#
#files = ['data/train.txt', 'data/test.txt', 'data/valid.txt']
# with open(eventTypesPath) as file:
#    eventTypes = [eventType.replace('\n', '')
#                  for eventType in file.readlines()]
#
# with open(eventSubtypesPath) as file:
#    eventSubtypes = [eventSubtype.replace(
#        '\n', '') for eventSubtype in file.readlines()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trigger pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('--embeddings', default='google',
                        help='Embeddings type: google, glove, fasttext, numberbatch, bert')
    parser.add_argument('--train', default='data/train.txt',
                        help='Train file')
    parser.add_argument('--test', default='data/test.txt',
                        help='Test file')
    parser.add_argument('--valid', default='data/valid.txt',
                        help='Validation file')
    parser.add_argument('--augment', default=False,
                        help='Augment or not')

    args = parser.parse_args()

    files = [args.train, args.test, args.valid]
#    files = [args.valid]

    assert os.path.exists(args.train)
    assert os.path.exists(args.test)
    assert os.path.exists(args.valid)

    logger.info('Loading pretrained embeddings {}'.format(args.embeddings))

    embeddingsPklPath = 'data/processed/embeddings_' + args.embeddings + '.pkl.gz'

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
        except:
            vocab = embeddings_model.vocab
    
        try:
            size_embeddings = embeddings_model.shape[1]
        except:
            size_embeddings = embeddings_model.vector_size

    embeddings, word2Idx, pos2Idx, idx2Pos, words, idx2word, char2Idx, idx2Char = \
        [], {}, {}, {}, {}, {}, {}, {}

    vocab_in_text = {}
    for fileIdx in range(len(files)):
        already_parsed = []
        file = files[fileIdx]
        logger.info('Pre-processing {}'.format(file))

        df = pd.read_csv(file, sep='\t', names=['doc_id',  'type', 'subtype', 'blabla',
                                                'sent', 'anchor_text', 'anchors'])

        for idx, line in tqdm(df.iterrows(), total=len(df)):
            sentence = line.sent
            doc = nlp(sentence)

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
#                    word2Idx["PADDING"] = len(word2Idx)
#                    # Zero vector vor 'PADDING' word
                    
                    
                    if 'bert' not in args.embeddings:
                        if 'PADDING' in embeddings_model:
                            embeddings.append(embeddings_model['PADDING'])
                        else:
                            vector = np.zeros(size_embeddings)
                            embeddings.append(vector)
                        vector = np.random.normal(loc=emb_mean, scale=emb_std, size=size_embeddings)
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
                        
                        vector = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), size_embeddings)
                        embeddings.append(vector)
                        word2Idx["UNKNOWN"] = len(word2Idx)
                        idx2word[word2Idx["UNKNOWN"]] = "UNKNOWN"


#                    word2Idx["UNKNOWN"] = len(word2Idx)
#                    #vector = np.random.uniform(-max_embed, max_embed, size_embeddings)
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
#                try:
#                    doc = nlp(sentence)
#                except:
                                    
                if len(word2Idx) != len(idx2word):
                    
                    import pdb
                    pdb.set_trace()

                tokens = [x for x in doc]
                
                if 'bert' in args.embeddings:
                  
#                    doc = nlp(sentence)
#                    clean_tokens = [x.text for x in clean_text(sentence)]

                    sentence = Sentence(' '.join([x.text for x in tokens]))
                    bert_embedding.embed(sentence)#3072
                    for token in sentence:
                        if token.text not in word2Idx:
                            embeddings.append(token.embedding.cpu().numpy())
                            word2Idx[token.text] = len(word2Idx)
                            idx2word[word2Idx[token.text]] = token.text
                            words[token.text] = True
                            if token.text in vocab_in_text:
                                vocab_in_text[token.text] += 1
                            else:
                                vocab_in_text[token.text] = 1
                            for char in token.text:
                                if char not in char2Idx:
                                    char2Idx[char] = len(char2Idx)
                                    idx2Char[char2Idx[char]] = char

#                    print(len(idx2word),len(word2Idx))
                else:
                    for idx, word in enumerate(tokens):
                        token = word.text
                        lemma = word.lemma_
                        
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
                                elif lemma in vocab:
                                    embeddings.append(embeddings_model[lemma])
                                    word2Idx[token] = len(word2Idx)
                                    idx2word[word2Idx[token]] = token
                                    words[token] = True
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
    if 'bert' not in args.embeddings:
        for word in vocab:
            if word not in word2Idx:
                unknown_words += 1

    logger.info('Generating embeddings for unknown words. Known words: {} Unknown words: {}'.format(
        len(word2Idx), unknown_words))
#    add_unknown_words(embeddings, words, word2Idx, idx2word, vocab_in_text,
#                      k=size_embeddings, mean=emb_mean, std=emb_std)

    maxSentenceLen = [57]

    train_set = create_matrices(files[0], word2Idx, max(maxSentenceLen),
                                char2Idx, idx2Char, args.augment, args.embeddings)
    f = gzip.open('data/processed/triggers_' +
                  args.embeddings + '_train.pkl.gz', 'wb')
    pkl.dump(train_set, f, -1)
    f.close()

    valid_set = create_matrices(files[2], word2Idx, max(maxSentenceLen),
                                char2Idx, idx2Char, args.augment, args.embeddings)
    f = gzip.open('data/processed/triggers_' +
                  args.embeddings + '_valid.pkl.gz', 'wb')
    pkl.dump(valid_set, f, -1)
    f.close()

    test_set = create_matrices(files[1], word2Idx, max(maxSentenceLen),
                               char2Idx, idx2Char, args.augment, args.embeddings)
    f = gzip.open('data/processed/triggers_' +
                  args.embeddings + '_test.pkl.gz', 'wb')
    pkl.dump(test_set, f, -1)
    f.close()
    
    logger.info('Saving processed data in data/processed')
    word2IdxPklPath = "data/processed/word2Idx_"+args.embeddings+".pkl.gz"
    f = gzip.open(word2IdxPklPath, 'wb')
    pkl.dump(word2Idx, f, -1)
    f.close()

    idx2wordPklPath = "data/processed/idx2word_"+args.embeddings+".pkl.gz"
    f = gzip.open(idx2wordPklPath, 'wb')
    pkl.dump(idx2word, f, -1)
    f.close()
    
    print(len(idx2word),len(word2Idx))
    
    char2idxPklPath = "data/processed/char2idx_"+args.embeddings+".pkl.gz"
    f = gzip.open(char2idxPklPath, 'wb')
    pkl.dump(char2Idx, f, -1)
    f.close()

    idx2charPklPath = "data/processed/idx2char_"+args.embeddings+".pkl.gz"
    f = gzip.open(idx2charPklPath, 'wb')
    pkl.dump(idx2Char, f, -1)
    f.close()

    f = gzip.open(embeddingsPklPath, 'wb')
    pkl.dump(embeddings, f, -1)
    f.close()
    
    if 'bert' not in args.embeddings:
        vocab_metadata = 'data/vocab.txt'
        f_vocab = open(vocab_metadata, 'w')
        for i, j in idx2word.items():
            f_vocab.write(j + '\n')
        f_vocab.close()

#    f = gzip.open(embeddingsPklPath, 'rb')
#    embeddings = pkl.load(f)
#    embeddings = np.array(embeddings)
#    f.close()
#
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



#    print(idx2Char)
    #print('VOCAB', len(vocab))
    logger.info('Maximum sentence length {}'.format(maxSentenceLen))
    logger.info('Total words {}'.format(len(word2Idx)))
