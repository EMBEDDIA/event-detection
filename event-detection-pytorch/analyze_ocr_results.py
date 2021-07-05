# -*- coding: utf-8 -*-
import pandas as pd
import argparse
from ast import literal_eval
import spacy
import MicroTokenizer

#RANDOM = 8689
RANDOM = 567567

multilingual_nlp = spacy.load('xx_ent_wiki_sm')
multilingual_nlp.add_pipe(multilingual_nlp.create_pipe('sentencizer'))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trigger pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('--data_path', 
                        help='Embeddings type: google, glove, fasttext and numberbatch')

    args = parser.parse_args()

    df = pd.read_csv(args.data_path, sep='\t', names=['doc_id',  'type', 'subtype', 'old_index',
                                            'sent', 'anchor_text', 'anchors', 'language'],
                     converters={"anchors": literal_eval})#.sample(frac=.1) #TODO


    affected_events = 0
    total_events = 0
    for idx, line in df.iterrows():
        sentence = line.sent
        language = line.language
        
#        if len(line.anchors) > 0:
#            print(line.anchors)
        for anchor in line.anchors:
            total_events += 1
            text_anchor = anchor[0]

            start, end = anchor[2].split(',')
            
            doc = multilingual_nlp(sentence)
            

            if language == 'zh':
                tokens = MicroTokenizer.cut(
                    sentence, HMM=True)
                tokens = [x.lower() for x in tokens]
            else:
                tokens = [x.text.lower() for x in doc]
            
            if ' ' not in text_anchor:
                if text_anchor.lower() not in tokens:
                    affected_events += 1
                    print(text_anchor, '----', tokens)#, '---', anchor[2], '----', sentence.lower().index(text_anchor.lower()))
#                print(tokens)
            else:
#                print()
                if text_anchor.lower() not in sentence.lower():
                    affected_events += 1
                    print(text_anchor)
                    print(sentence)
                    print('--'*30)
    
    print((affected_events * 100.0)/total_events, '% affected triggers')
                
                