# Event Detection

In Natural Language Processing, an event denotes a change in states and is usually described by a set of participants(i.e. attributes, roles) doing actions. With each event comes an event trigger(s) which can be seen as what invokes the event. These triggers more than often can be seen in the form of verbs and nouns, while sometimes can be expressed through adverbs and pronouns. 

Identifying event triggers and classifying them into their respective categories are called Event Detection. It's a challenging subtask of Event Extraction since 1 event can have multiple forms of event triggers, and vice versa, the same event trigger might lead to different events depending on the context.

# Data Analysis for Information Extraction in any Languages (DAniEL)

In the context of the NewsEye project where many languages are considered, the DAniEL system, created by Gael Lejeurn, was chosen. The system focuses on Epidemic Surveillance over press articles across multiple languages. Rather depending on language-specific grammars and analyzers, the system implements a string-based algorithm that detects repeated maximal substrings in salient zones to extract important information (disease name, location). The decision of choosing prominent text zones is based on the characteristics of a journalistic writing style where crucial information is usually put at the beginning and end of the article.

This is the Python 3 version of the Daniel System (originally from here: https://github.com/rundimeco/daniel)

## Prerequisites

* Python3.x (we are currently using version 3.7)
* Justtext Python module (for cleaning up HTML boilerplates if needed)

## Usage

DAniEL can be used to check a single file or to handle big corpus

### Testing a single file

    python3 daniel.py -l LANGUAGE_ID -d PATH_TO_DOCUMENT -v

### Processing a corpus

    python3 process_corpus.py -c PATH_TO_CORPUS_FILE [-r RATIO] 

### Evaluate the Result of corpus processing

    python3 evaluate.py PATH_TO_GROUNDTRUTH PATH_TO_RESULT_FILE

or simply adding "-e" before running process_corpus.py will do

## Annotating Guide

The corpus is annotated using JSON format. It's basically a dictionary where the key is the document's ID. The value corresponding to a key is as follow:
- Mandatory Value:
    * Document's Path
- Useful Value:
    * Document's Source
    * Language
    * URL
    * Comment
- Annotation:
    * Pair Value of [Disease Name, Location]

An example can be found in docs/Indonesian_GL.json

# Dataset

To the best of found knowledge, there is no public corpus that has both DAniEL-compatible annotations and OCR noise. The available corpora only have either annotations that work with DAniEL nor noisy text that was generated during the OCR process. To deal with this problem, we built a noisy dataset based on previously-available DAniEL corpora.

The dataset can be found [here](https://1drv.ms/u/s!At3pWQFublT-vEDzFSvaW0qCevjk?e=RLCfym), including both the degraded images and the noisy text extracted using OCR, with 2 noise types: Character Degradation and Phantom Character. There are 6 languages in this corpus: English, French, Russian, Polish, Chinese and Greek.
