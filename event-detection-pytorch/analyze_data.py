# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import json
import os
import re
from nltk.probability import FreqDist
import spacy
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pprint import pprint
from nltk import bigrams
import MicroTokenizer

#RANDOM = 8689
RANDOM = 567567

multilingual_nlp = spacy.load('xx_ent_wiki_sm')
multilingual_nlp.add_pipe(multilingual_nlp.create_pipe('sentencizer'))


def chinese_sentence_tokenizer(text):
    sentences = []
    for sent in re.findall(u'[^!?？。\.\!\?\？]+[!?。\.\!\?]?', text, flags=re.U):
        sentences.append(sent)
    return sentences


def format_number(number):
    return "{:,}".format(int(number))


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch Event Detection')
    parser.add_argument("--input_directory", default="../../data/data_daniel/text/",
                        help="JSON file for the corpus to process")
    args = parser.parse_args()

    input_directory = args.input_directory
    assert os.path.exists(input_directory)

    corpora = {}
    for path, directories, files in os.walk(input_directory):
        for file in files:
            if file.endswith('json'):
                if 'all' not in file and 'train' not in file and 'test' not in file and 'valid' not in file:
                    file_path = os.path.join(path, file)
                    language = file.replace('.json', '').split('_')[1]
                    print('File: %s, language: %s' %
                          (os.path.join(path, file), language))
                    corpora[language] = file_path

    freq_dist = FreqDist()

    all_data = []
    all_ids = []
    all_languages = []
    docs = []
    number_annotations = []

    languages = []
    docs_with_events = []
    docs_per_language = {}
    docs_per_language_with_events = {}

    print('Generating data:')
    annotations_found = 0
    for language, corpus_path in tqdm(corpora.items(), total=len(corpora.items())):
        with open(corpus_path) as json_file:
            data = json.load(json_file)
            print(corpus_path)
            for key, value in data.items():
                annotations = [x for x in value['annotations']
                               [0] if x not in ['N', 'unknown', '', ' ']]
                number_annotations.append(len(annotations))
                if len(annotations) > 0:
                    docs_with_events.append(1)
                else:
                    docs_with_events.append(0)

                document_path = os.path.join(
                    input_directory, value['document_path'])
                with open(document_path, 'r') as r:
                    soup = BeautifulSoup(r.read(), features="lxml")
                    text = soup.get_text()

                text = re.sub('\n+', '\n', text)
                text = re.sub('\s+', ' ', text)
                text = re.sub('"', ' ', text)
                text = re.sub("'", ' ', text).strip()

                new_annotations = []
 #               if len(annotations) > 0:
 #                   for idx, annotation in enumerate(annotations):
 #                       original_annotation = annotation
 #                       if len(annotation) < 4:
 #                           continue
 #                       if annotation.isdigit():
 #                           annotation = format_number(annotation)
 #                           new_annotations.append(annotation)
 
 #
 #                           text = text.replace(annotation, original_annotation)
 #                           
 #                           new_annotations.append(annotation.replace(',', '.'))
 #                           
 #                           text = text.replace(annotation.replace(',', '.'), original_annotation)
 #                           
 #                           if annotation.count(',') == 1:
 #                               new_annotations.append(annotation.replace(
 #                                   annotation[annotation.index(','):], ' tys'))
 #                               text = text.replace(annotation.replace(
 #                                   annotation[annotation.index(','):], ' tys'), original_annotation)

#                        if language == 'ru':
#                            new_annotation = annotation
#                            if 'ай' in annotation:
#                                new_annotations.append(annotation.replace('ай', 'ае'))
                                
#                                text = text.replace(annotation.replace('ай', 'ае'), original_annotation)
                                
#                                new_annotation = new_annotation.replace('ай', 'ае')
#                            if 'ий' in annotation:
#                                new_annotations.append(annotation.replace('ий', 'ьего'))
                                
#                                text = text.replace(annotation.replace('ий', 'ьего'), original_annotation)
                                
#                                new_annotation = new_annotation.replace('ий', 'ьего')
#                            if 'ипп' in annotation:#птичьего грипп птичьего гриппа
#                                new_annotations.append(annotation.replace('ипп', 'иппа'))
                                
#                                text = text.replace(annotation.replace('ипп', 'иппа'), original_annotation)
                                
#                                new_annotation = new_annotation.replace('ипп', 'иппа')
#                            if 'рипп' in annotation:#птичьего грипп
#                                new_annotations.append(annotation.replace('рипп', 'риппа'))
                                
#                                text = text.replace(annotation.replace('рипп', 'риппа'), original_annotation)
                                
#                                new_annotation = new_annotation.replace('рипп', 'риппа')
#                            if 'во' in annotation:#бешенством бешенством бешенством
#                                new_annotations.append(annotation.replace('во', 'вом'))
                                
#                                text = text.replace(annotation.replace('во', 'вом'), original_annotation)
                                
#                                new_annotation = new_annotation.replace('во', 'вом')
#                            if 'ир' in annotation:#мире мир
#                                new_annotations.append(annotation.replace('ир', 'ире'))
                                
#                                text = text.replace(annotation.replace('ир', 'ире'), original_annotation)
                                
#                                new_annotation = new_annotation.replace('ир', 'ире')

#                                text = text.replace(new_annotation, original_annotation)

#                            new_annotations.append(new_annotation)
                            
#                            text = text.replace(new_annotation, original_annotation)

                doc = multilingual_nlp(text)

                sentences = [
                    sent for sent in doc.sents if len(sent.text) > 3]

                if language == 'zh':
                    class ChineseSentence:
                        def __init__(self, text):
                            self.text = text
                    sentences = [ChineseSentence(
                        x) for x in chinese_sentence_tokenizer(text)]

                docs.append(key)
                languages.append(language)


                if len(annotations) > 0:
                    
                    print('-' * 50)
                    print('Initial:', annotations)
                    for annotation in annotations:

                        if annotation not in ['N', 'unknown', ' ']:
                            found = False
#                            if annotation.isdigit():
#                                annotation = format_number(annotation)
                            for sentence in sentences:
                                if language == 'zh':
                                    sentence = sentence.text
                                    tokens = MicroTokenizer.cut(
                                        sentence, HMM=True)
                                    tokens_lower = MicroTokenizer.cut(
                                        sentence, HMM=True)
                                else:
                                    tokens_lower = [x.text.lower()
                                                    for x in sentence]
                                    tokens = [x.text for x in sentence]

                                ngrams = [' '.join(x)
                                          for x in list(bigrams(tokens))]
                                ngrams_lower = [' '.join(x) for x in list(
                                    bigrams(tokens_lower))]

                                tokens_lower += tokens + ngrams + ngrams_lower

                                if not language == 'zh':
                                    sentence = sentence.text

                                if annotation.lower() in tokens_lower:
                                    try:
                                        index = tokens.index(annotation)
                                    except:
                                        index = tokens_lower.index(
                                            annotation.lower())
                                        sentence = sentence.replace(annotation.lower(), annotation)
#                                        annotation = annotation.lower()

                                    print(annotation, index)
                                    annotations_found += 1
#                                    print(tokens_lower)

                                    found = True
                                    line = key + '\tEvent\tEvent\t1\t' + sentence + \
                                        '\t' + annotation + '\t' + str(
                                            [(annotation, 'Event', str(index) + ',' +
                                              str(index+len(annotation)), 'EV')]) + '\t' + language
                                    all_ids.append(key)
                                    all_data.append(line)
                                    all_languages.append(language)
                                    if language not in docs_per_language:
                                        docs_per_language[language] = []
                                    else:
                                        docs_per_language[language].append(
                                            (key, line))

                    for sentence in sentences:
                        no_annotation = True

                        if language == 'zh':
                            sentence = sentence.text
                            tokens = MicroTokenizer.cut(sentence, HMM=True)
                            tokens_lower = MicroTokenizer.cut(
                                sentence, HMM=True)
                        else:
                            tokens_lower = [x.text.lower() for x in sentence]
                            tokens = [x.text for x in sentence]

                        ngrams = [' '.join(x) for x in list(bigrams(tokens))]
                        ngrams_lower = [' '.join(x)
                                        for x in list(bigrams(tokens_lower))]

                        tokens_lower += tokens + ngrams + ngrams_lower

                        if not language == 'zh':
                            sentence = sentence.text

                        for annotation in annotations:
                            if annotation in tokens:
                                no_annotation = False
                            if annotation.lower() in tokens_lower:
                                no_annotation = False
                        if no_annotation == True:
                            line = key + '\tEvent\tEvent\t1\t' + \
                                sentence + '\tNone\t' + \
                                str([]) + '\t' + language
                            all_data.append(line)
                            all_ids.append(key)
                            all_languages.append(language)
                            if language not in docs_per_language:
                                docs_per_language[language] = []
                            else:
                                docs_per_language[language].append((key, line))

                elif len(annotations) == 0:
                    for sentence in sentences:
                        sentence = sentence.text
                        line = key + '\tEvent\tEvent\t1\t' + \
                            sentence + '\tNone\t' + str([]) + '\t' + language
                        all_data.append(line)
                        all_ids.append(key)
                        all_languages.append(language)
                        if language not in docs_per_language:
                            docs_per_language[language] = []
                        else:
                            docs_per_language[language].append((key, line))

    train_docs, test_docs, train_langs, test_langs, \
        train_number_annotations, test_number_annotations, \
        train_docs_with_events, test_docs_with_events = \
        train_test_split(docs, languages, number_annotations, docs_with_events,
                         test_size=0.2, random_state=RANDOM, stratify=languages,
                         shuffle=True)

    test_docs, val_docs, test_langs, val_langs, \
        test_number_annotations, val_number_annotations, \
        test_docs_with_events, val_docs_with_events = \
        train_test_split(test_docs, test_langs, test_number_annotations,
                         test_docs_with_events, test_size=0.5,
                         random_state=RANDOM,
                         stratify=test_langs,
                         shuffle=True)
    
    import numpy as np
    for lang in list(set(train_langs)):
        indices = [i for i, x in enumerate(train_langs) if x == lang]
        print('Train', lang, len([x for x in np.array(train_number_annotations)[indices] if x > 0]))
        indices = [i for i, x in enumerate(val_langs) if x == lang]
        print('Validation', lang, len([x for x in np.array(val_number_annotations)[indices] if x > 0]))
        indices = [i for i, x in enumerate(test_langs) if x == lang]
        print('Test', lang, len([x for x in np.array(test_number_annotations)[indices] if x > 0]))
        print('-'*20)
        
    print('Train, test, validation')
    print(len(train_docs), len(test_docs), len(val_docs))

    print('Train:')
    pprint(FreqDist(train_langs))
    print('Test:')
    pprint(FreqDist(test_langs))
    print('Validation:')
    pprint(FreqDist(val_langs))

    print('Train:', sum(train_number_annotations))
    print('Test:', sum(test_number_annotations))
    print('Validation:', sum(val_number_annotations))
    print('Annotations found:', annotations_found)
    import pdb;pdb.set_trace()
    print(train_number_annotations)
    print(train_number_annotations)
    print(train_number_annotations)
    train_number_annotations = dict(FreqDist(train_number_annotations))
    test_number_annotations = dict(FreqDist(test_number_annotations))
    val_number_annotations = dict(FreqDist(val_number_annotations))
    del train_number_annotations[0]
    del test_number_annotations[0]
    del val_number_annotations[0]

    print('Train (documents with events):', sum(train_number_annotations.values(
    )), (sum(train_number_annotations.values())*100.0)/len(train_docs))
    print('Test (documents with events):', sum(test_number_annotations.values()),
          (sum(test_number_annotations.values())*100.0)/len(test_docs))
    print('Validation (documents with events):', sum(val_number_annotations.values(
    )), (sum(val_number_annotations.values())*100.0)/len(val_docs))

    # 3857 482 483
    print('Generating splits per language:')
#    for idx_language, items in tqdm(docs_per_language.items(), total=len(docs_per_language.items())):
    for idx_language in set(languages):

        data_path = os.path.join(input_directory, idx_language)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        file_name = os.path.join(data_path, 'test.txt')
        with open(file_name, 'w') as file:
            indices = [i for i, x in enumerate(
                all_languages) if x == idx_language]
            for i in indices:
                file.write(all_data[i] + '\n')

        file_name = os.path.join(
            data_path, 'test_ids.txt')
        with open(file_name, 'w') as file:
            indices = [i for i, x in enumerate(
                all_languages) if x == idx_language]
            for i in indices:
                file.write(all_ids[i] + '\n')

        file_name = os.path.join(
            data_path, 'test_documents.txt')
        with open(file_name, 'w') as file:
            indices = [i for i, x in enumerate(
                all_languages) if x == idx_language]
            for i in set([all_ids[j] for j in indices]):
                file.write(i + '\n')

        data_json = {}
        file_name = os.path.join(
            input_directory, 'daniel_test_' + idx_language + '.json')

        indices = [i for i, x in enumerate(all_languages) if x == idx_language]
        with open(corpora[idx_language]) as json_file:
            data = json.load(json_file)
            for key, value in data.items():
                if key in set([all_ids[j] for j in indices]):
                    data_json[key] = value

        with open(file_name, 'w') as outfile:
            json.dump(data_json, outfile, ensure_ascii=True,
                      indent=4, sort_keys=True)

        other_data = [x for i, x in enumerate(
            all_data) if idx_language != all_languages[i]]
        other_indices = [x for i, x in enumerate(
            all_ids) if idx_language != all_languages[i]]

        other_docs = [x for i, x in enumerate(
            docs) if idx_language != languages[i]]
        other_languages = [x for i, x in enumerate(
            languages) if idx_language != languages[i]]

        print(len(other_languages), len(other_data),
              len(other_docs), len(other_indices))

        train_docs_per_language, val_docs_per_language, train_langs_per_language, val_langs_per_language = \
            train_test_split(other_docs, other_languages,
                             test_size=0.2, random_state=RANDOM,
                             stratify=other_languages, shuffle=True)
        #import pdb;pdb.set_trace()
        
        print('train:', len(train_docs_per_language),
              'valid:', len(val_docs_per_language))

        for entry in [('train', train_docs_per_language),
                      ('valid', val_docs_per_language)]:

            print('Generating', entry[0], 'with test on', idx_language)

            file_name, documents = os.path.join(
                data_path, entry[0] + '.txt'), entry[1]
            with open(file_name, 'w') as file:
                for doc in documents:
                    indices = [i for i, x in enumerate(other_indices) if x == doc]
                    for i in indices:
                        file.write(other_data[i] + '\n')

            file_name = os.path.join(
                data_path, entry[0] + '_ids.txt')
            with open(file_name, 'w') as file:
                for key in documents:
                    file.write(str(key) + '\n')

            file_name, documents = os.path.join(
                data_path, entry[0] + '_documents.txt'), entry[1]
            with open(file_name, 'w') as file:
                for doc in set(documents):
                    file.write(doc + '\n')

            print('Generating json', entry[0], idx_language)
            data_json = {}
            file_name, ids = os.path.join(
                input_directory, 'daniel_' + entry[0] + '_' + idx_language + '.json'), entry[1]
            for language, corpus_path in corpora.items():
                with open(corpus_path) as json_file:
                    data = json.load(json_file)
                    for key, value in data.items():
                        if key in documents:
                            data_json[key] = value

            with open(file_name, 'w') as outfile:
                json.dump(data_json, outfile, ensure_ascii=True,
                          indent=4, sort_keys=True)

    data_path = os.path.join(input_directory, 'all')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for entry in [('train', train_docs), ('test', test_docs), ('valid', val_docs)]:
        file_name, documents = os.path.join(
            data_path, entry[0] + '.txt'), entry[1]

        with open(file_name, 'w') as file:
            for doc in documents:
                indices = [i for i, x in enumerate(all_ids) if x == doc]
                for i in indices:
                    file.write(all_data[i] + '\n')

    for entry in [('test', test_docs), ('valid', val_docs)]:
        file_name, documents = os.path.join(
            data_path, entry[0] + '_ids.txt'), entry[1]
        with open(file_name, 'w') as file:
            for doc in documents:
                indices = [i for i, x in enumerate(all_ids) if x == doc]
                for i in indices:
                    file.write(all_ids[i] + '\n')

    for entry in [('test', test_docs), ('valid', val_docs)]:
        file_name, documents = os.path.join(
            data_path, entry[0] + '_documents.txt'), entry[1]
        with open(file_name, 'w') as file:
            for doc in documents:
                file.write(doc + '\n')

    test_ids = []
    with open(os.path.join(data_path,  'test_documents.txt'), 'r') as f:
        for _id in f.readlines():
            test_ids.append(_id.replace('\n', '').split('/')[-1])
    valid_ids = []
    with open(os.path.join(data_path, 'valid_documents.txt'), 'r') as f:
        for _id in f.readlines():
            valid_ids.append(_id.replace('\n', '').split('/')[-1])

    for entry in [('test', test_ids), ('valid', valid_ids)]:
        data_json = {}
        file_name, ids = os.path.join(
            input_directory, 'daniel_' + entry[0] + '_all.json'), entry[1]
        for language, corpus_path in corpora.items():
            with open(corpus_path) as json_file:
                data = json.load(json_file)
                for key, value in data.items():
                    if key in ids:
                        data_json[key] = value

        with open(file_name, 'w') as outfile:
            json.dump(data_json, outfile, ensure_ascii=True,
                      indent=4, sort_keys=True)


#    for label, freq in freq_dist.most_common(100):
#        print("%s : %f%%" % (label, 100*freq / float(freq_dist.N())))
#    freq_dist.plot(30, cumulative=False)
