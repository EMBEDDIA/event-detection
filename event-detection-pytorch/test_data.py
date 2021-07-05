# -*- coding: utf-8 -*-
import pandas as pd
from ast import literal_eval
from nltk.probability import FreqDist

df = pd.read_csv('data/valid.txt', sep='\t', names=['doc_id',  'type', 'subtype', 'old_index',
                                        'sent', 'anchor_text', 'anchors'],
                 converters={"anchors": literal_eval})
df_ace = pd.read_csv('data/valid_ace2.txt', sep='\t', names=['doc_id',  'type', 'subtype', 'old_index',
                                        'sent', 'anchor_text', 'anchors'],
                 converters={"anchors": literal_eval})

fdist = FreqDist()
for idx, element in df.iterrows():
    anchors = element.anchors
    for anchor in anchors:
#        print(anchor[0])
        fdist[anchor[0]] += 1

for label, freq in fdist.most_common(100):
    print("%s : %f%%" % (label, freq))#100*freq / float(fdist.N())))

print('-'*20)
fdist_ace = FreqDist()
for idx, element in df_ace.iterrows():
    anchors = element.anchors
    for anchor in anchors:
        fdist_ace[anchor[0]] += 1
        
for label, freq in fdist_ace.most_common(100):
    print("%s : %f%%" % (label, freq))#100*freq / float(fdist_ace.N())))

import pdb;pdb.set_trace()