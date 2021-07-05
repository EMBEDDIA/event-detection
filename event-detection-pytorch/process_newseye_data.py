# -*- coding: utf-8 -*-

#[{"id"=>"arbeiter_zeitung_aze18950418_article_420", "all_text_tde_siv"=>"Vereinsabe. Mahlverein, Grohgasse 5. Jeden Montag\nVereinrabend, eventuell Diskussion.\nV. Verein der in der gesammten Napiorbranche beschäftigten\nArbeiter und Kromkkeinnen Niederosterr., Vereinslokal Zimmer¬\nmann's Gasthaus, V. Ziegelosengasse 23.", "date_created_dtsi"=>"1895-04-18T00:00:00Z"}
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import json
import os, re
from nltk.probability import FreqDist
import spacy
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

multilingual_nlp = spacy.load('xx_ent_wiki_sm')
multilingual_nlp.add_pipe(multilingual_nlp.create_pipe('sentencizer'))
###
# AFP_ENG_20030401.0476	Personnel	End-Position	26	
# former  senior banker Callum McCarthy begins what is one of the most important jobs in London 's financial world in September , when incumbent Howard Davies steps down .	
# steps down	[('steps down','End-Position','494,498','EV-4')]
###

if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch Event Detection')
    parser.add_argument("-d", "--directory", default="../../data_daniel/text/input",
                      help="JSON file for the corpus to process")
    args = parser.parse_args()
    
#    corpus_path = args.corpus
    directory = args.directory
#    assert os.path.exists(corpus_path)

    test_ids = []
    with open('../../data_daniel/test_documents.txt', 'r') as f:
        for _id in f.readlines():
            test_ids.append(_id.replace('\n', '').split('/')[-1])
#    import pdb;pdb.set_trace()
    corpora = {}
    results_json = {}
    for path, directories, files in os.walk(directory):
        for file in files:
            if file.endswith('json'):
                file_path = os.path.join(path, file)
                print(file_path)
                with open(file_path) as json_file:
                    data = json.load(json_file)
            for key, value in data.items():
                if key in test_ids:
                    results_json[key] = value

        with open('results/CNN_daniel_test.json', "w") as write_file:
            json.dump(results_json, write_file)

#                corpora[language] = file_path
