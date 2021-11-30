# -*- coding: utf-8 -*-
import os, re
import json

rootDir = 'output/'
#from transformers import pipeline
# Allocate a pipeline for sentiment-analysis
#classifier = pipeline('sentiment-analysis')
def clean_str(text):
#    text = text.lower()
    # Clean the text
#    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", " . ", text)
#    text = re.sub(r",", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
#    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace('- lrb -', '')
    text = text.replace('- rrb -', '')
    text = text.replace('- LRB -', '')
    text = text.replace('- RRB -', '')
    text = re.sub(r"\s+", ' ', text).strip()

    return text
ENTITY_MARKERS = [('[E11]', '[E12]'), ('[E21]', '[E22]'), ('[E11]', '[E12]'), ('[E21]', '[E22]'), ('[E11]', '[E12]'), ('[E21]', '[E22]'), ('[E11]', '[E12]'), ('[E21]', '[E22]')]


LABELS = ['Nominate', 'Start_Position', 'End_Position', 'Meet', 'Attack', 'Transfer_Money', 'Transport', 'Start_Org', 'Marry', 'Die', 
          'Phone_Write', 'Arrest_Jail', 'Elect', 'Convict', 'Sentence', 'Charge_Indict', 'Release_Parole', 'Appeal', 'Demonstrate', 
          'Trial_Hearing', 'Divorce', 'Sue', 'Transfer_Ownership', 'Fine', 'Acquit', 'Execute', 'End_Org', 'Declare_Bankruptcy', 
          'Be_Born', 'Extradite', 'Injure', 'Merge_Org', 'Pardon']

QUESTIONS = {
'Nominate': 'What is the nomination?', 
'Start_Position': 'What is the start position?', 
'End_Position': 'What is the end position?', 
'Meet': 'What is meeting?', 
'Attack': 'What is attacking?', 
'Transfer_Money': 'What is transferring money?', 
'Transport': 'What is transporting?', 
'Start_Org': 'What is start organization?', 
'Marry': 'What is marriage?', 
'Die': 'What is dying?', 
'Phone_Write': 'What is phone writing?', 
'Arrest_Jail': 'What is arresting jail?', 
'Elect': 'What is election?', 
'Convict': 'What is conviction?', 
'Sentence': 'What is sentencing?', 
'Charge_Indict': 'What is charge indicting?', 
'Release_Parole': 'What is releasing parole?',
'Appeal': 'What is appealing?',
'Demonstrate': 'What is demonstrating?',
'Trial_Hearing': 'What is trial hearing?',
'Divorce': 'What is divorcing?',
'Sue': 'Shat is sueing?',
'Transfer_Ownership': 'What is transferring ownership?',
'Fine': 'What is fining?',
'Acquit': 'What is acquitting?',
'Execute': 'What is executing?',
'End_Org': 'What is ending organization?',
'Declare_Bankruptcy': 'What is declaring bankrupcy?', 
'Be_Born': 'What is being born?',
'Extradite': 'What is extradition?',
'Injure': 'What is injuring?',
'Pardon': 'What is pardon?',
'Merge_Org': 'What is merging organization?'
}


#QUESTIONS = {
#'Nominate': 'What is nominate?', 
#'Start_Position': 'What is start position?', 
#'End_Position': 'What is end position?', 
#'Meet': 'What is meet?', 
#'Attack': 'What is attack?', 
#'Transfer_Money': 'What is transfer money?', 
#'Transport': 'What is transport?', 
#'Start_Org': 'What is start organization?', 
#'Marry': 'What is marry?', 
#'Die': 'What is die?', 
#'Phone_Write': 'What is phone write?', 
#'Arrest_Jail': 'What is arrest jail?', 
#'Elect': 'What is elect?', 
#'Convict': 'What is convict?', 
#'Sentence': 'What is sentence?', 
#'Charge_Indict': 'What is charge indict?', 
#'Release_Parole': 'What is release parole?',
#'Appeal': 'What is appeal?',
#'Demonstrate': 'What is demonstrate?',
#'Trial_Hearing': 'What is trial hearing?',
#'Divorce': 'What is divorce?',
#'Sue': 'What is sue?',
#'Transfer_Ownership': 'What is transfer ownership?',
#'Fine': 'What is fine?',
#'Acquit': 'What is acquit?',
#'Execute': 'What is execute?',
#'End_Org': 'What is end organization?',
#'Declare_Bankruptcy': 'What is declare bankruptcy?', 
#'Be_Born': 'What is be born?',
#'Extradite': 'What is extradite?',
#'Injure': 'What is injure?',
#'Pardon': 'What is pardon?',
#'Merge_Org': 'What is merge organization?'
#}
#
#QUESTIONS = {
#'Nominate': 'A NOMINATE Event occurs whenever a PERSON is proposed for a START-POSITION Event by the appropriate PERSON, through official channels. ', 
#'Start_Position': 'A START-POSITION Event occurs whenever a PERSON Entity begins working for (or changes offices within) an ORGANIZATION or GPE.  This includes government officials starting their terms, whether elected or appointed. ', 
#'End_Position': 'An END-POSITION Event occurs whenever a PERSON Entity stops working for (or changes offices within) an ORGANIZATION or GPE.  The change of office case will only be taggable when the office being left is explicitly mentioned within the scope of the Event.  This includes government officials ending terms, whether elected or appointed.  ', 
#'Meet': 'A MEET Event occurs whenever two or more Entities come together at a single location and interact with one another face-to-face.  MEET Events include talks, summits, conferences, meetings, visits, and any other Event where two or more parties get together at some location. ', 
#'Attack': 'An ATTACK Event is defined as a violent physical act causing harm or damage.  ATTACK Events include any such Event not covered by the INJURE or DIE subtypes, including Events where there is no stated agent.  The ATTACK Event type includes less specific violence-related nouns such as ‘conflict’, ‘clashes’, and ‘fighting’.  ‘Gunfire’, which has the qualities of both an Event and a weapon, should always be tagged as an ATTACK Event, if only for the sake of consistency.  A ‘coup’ is a kind of ATTACK (and so is a ‘war’).   ', 
#'Transfer_Money': 'TRANSFER-MONEY Events refer to the giving, receiving, borrowing, or lending money when it is not in the context of purchasing something.  The canonical examples are: (1) people giving money to organizations (and getting nothing tangible in return); and (2) organizations lending money to people or other orgs. ', 
#'Transport': 'A TRANSPORT Event occurs whenever an ARTIFACT (WEAPON or VEHICLE) or a PERSON is moved from one PLACE (GPE, FACILITY, LOCATION) to another.', 
#'Start_Org': 'A START-ORG Event occurs whenever a new ORGANIZATION is created.', 
#'Marry': 'MARRY Events are official Events, where two people are married under the legal definition.', 
#'Die': 'A DIE Event occurs whenever the life of a PERSON Entity ends.  DIE Events can be accidental, intentional or self-inflicted.  ', 
#'Phone_Write': 'A PHONE-WRITE Event occurs when two or more people directly engage in discussion which does not take place ‘face-to-face’.  To make this Event less open-ended, we limit it to written or telephone communication where at least two parties are specified.  Communication that takes place in person should be considered a MEET Event.  The very common ‘PERSON told reporters’ is not a taggable Event, nor is ‘issued a statement’.  A PHONE-WRITE Event must be explicit phone or written communication between two or more parties.', 
#'Arrest_Jail': 'A JAIL Event occurs whenever the movement of a PERSON is constrained by a state actor (a GPE, its ORGANIZATION subparts, or its PERSON representatives).    ', 
#'Elect': 'An ELECT Event occurs whenever a candidate wins an election designed to determine the PERSON argument of a START-POSITION Event. ', 
#'Convict': 'A CONVICT Event occurs whenever a TRY Event ends with a successful prosecution of the DEFENDANT-ARG.  In other words, a PERSON, ORGANIZATION or GPE Entity is convicted whenever that Entity has been found guilty of a CRIME. It can have a CRIME attribute filled by a string from the text.  CONVICT Events will also include guilty pleas.', 
#'Sentence': 'A SENTENCE Event takes place whenever the punishment (particularly incarceration) for the DEFENDANT-ARG of a TRY Event is issued by a state actor (a GPE, an ORGANIZATION subpart or a PERSON representing them).  It can have a CRIME-ARG attribute filled by a CRIME Value and a SENTENCE-ARG attribute filled by a SENTENCE Value.', 
#'Charge_Indict': 'A CHARGE Event occurs whenever a PERSON, ORGANIZATION or GPE is accused of a crime by a state actor (GPE, an ORGANIZATION subpart of a GPE or a PERSON representing a GPE).', 
#'Release_Parole': 'A RELEASE Event occurs whenever a state actor (GPE, ORGANIZATION subpart, or PERSON representative) ends its custody of a PERSON Entity.   This can be because the sentence has ended, because the charges are dropped, or because parole has been granted.',
#'Appeal': 'An APPEAL Event occurs whenever the decision of a court is taken to a higher court for review.',
#'Demonstrate': 'A DEMONSRATE Event occurs whenever a large number of people come together in a public area to protest or demand some sort of official action. DEMONSTRATE Events include, but are not limited to, protests, sit-ins, strikes, and riots. ',
#'Trial_Hearing': 'A TRIAL Event occurs whenever a court proceeding has been initiated for the purposes of determining the guilt or innocence of a PERSON, ORGANIZATION or GPE accused of committing a crime.  ',
#'Divorce': 'A DIVORCE Event occurs whenever two people are officially divorced under the legal definition of divorce.  We do not include separations or church annulments.',
#'Sue': 'A SUE Event occurs whenever a court proceeding has been initiated for the purposes of determining the liability of a PERSON, ORGANIZATION or GPE accused of committing a crime or neglecting a commitment.  It can have a CRIME attribute filled by a string from the text.   It is not important that the PLAINTIFF-ARG be a state actor (a GPE, an ORGANIZATION subpart or a PERSON representing them).',
#'Transfer_Ownership': 'TRANSFER-OWNERSHIP Events refer to the buying, selling, loaning, borrowing, giving, or receiving of artifacts or organizations. ',
#'Fine': 'A FINE Event takes place whenever a state actor issues a financial punishment to a GPE, PERSON or ORGANIZATION Entity, typically as a result of court proceedings.  It can have a CRIME attribute filled by a string from the text. ',
#'Acquit': 'An ACQUIT Event occurs whenever a trial ends but fails to produce a conviction.  This will include cases where the charges are dropped by the PROSECUTOR-ARG. ',
#'Execute': 'An EXECUTE Event occurs whenever the life of a PERSON is taken by a state actor (a GPE, its ORGANIZATION subparts, or PERSON representatives).  It can have a CRIME attribute filled by a string from the text.',
#'End_Org': 'An END-ORG Event occurs whenever an ORGANIZATION ceases to exist (in other words ‘goes out of business’). ',
#'Declare_Bankruptcy': 'A DECLARE-BANKRUPTCY Event will occur whenever an Entity officially requests legal protection from debt collection due to an extremely negative balance sheet. ', 
#'Be_Born': 'A BE-BORN Event occurs whenever a PERSON Entity is given birth to.  Please note that we do not include the birth of other things or ideas.',
#'Extradite': 'An EXTRADITE Event occurs whenever a PERSON is sent by a state actor from one PLACE (normally the GPE associated with the state actor, but sometimes a FACILITY under its control) to another place (LOCATION, GPE or FACILITY) for the purposes of legal proceedings there. ',
#'Injure': 'An INJURE Event occurs whenever a PERSON Entity experiences physical harm.  INJURE Events can be accidental, intentional or self-inflicted.',
#'Pardon': 'A PARDON Event occurs whenever a head-of-state or their appointed representative lifts a sentence imposed by the judiciary. ',
#'Merge_Org': 'A MERGE-ORG Event occurs whenever two or more ORGANIZATION Entities come together to form a new ORGANIZATION Entity.  This Event applies to any kind of ORGANIZATION, including government agencies.  It also includes joint ventures.'
#}

QUESTIONS = {
'Nominate': 'Nominate', 
'Start_Position': 'Start position', 
'End_Position': 'End position', 
'Meet': 'Meet', 
'Attack': 'Attack', 
'Transfer_Money': 'Transfer money', 
'Transport': 'Transport', 
'Start_Org': 'Start organization', 
'Marry': 'Marry', 
'Die': 'Die', 
'Phone_Write': 'Phone write', 
'Arrest_Jail': 'Arrest jail', 
'Elect': 'Elect', 
'Convict': 'Convict', 
'Sentence': 'Sentence', 
'Charge_Indict': 'Charge indict', 
'Release_Parole': 'Release parole',
'Appeal': 'Appeal',
'Demonstrate': 'Demonstrate',
'Trial_Hearing': 'Trial hearing',
'Divorce': 'Divorce',
'Sue': 'Sue',
'Transfer_Ownership': 'Transfer ownership',
'Fine': 'Fine',
'Acquit': 'Acquit',
'Execute': 'Execute',
'End_Org': 'End organization',
'Declare_Bankruptcy': 'Declare bankruptcy', 
'Be_Born': 'Be born',
'Extradite': 'Extradite',
'Injure': 'Injure',
'Pardon': 'Pardon',
'Merge_Org': 'Merge organization'
}


QUESTIONS = {
'Nominate': 'Nominate person agent position time place', 
'Start_Position': 'Start position person entity position time place', 
'End_Position': 'End position person entity position time place', 
'Meet': 'Meet entity time place', 
'Attack': 'Attack attacker target instrument time place', 
'Transfer_Money': 'Transfer money giver recipient beneficiary money time place', 
'Transport': 'Transport agent artifact vehicle price origin destination time', 
'Start_Org': 'Start organization agent organization time place', 
'Marry': 'Marry person time place', 
'Die': 'Die agent victim instrument time place', 
'Phone_Write': 'Phone write entity time', 
'Arrest_Jail': 'Arrest jail person agent crime time place', 
'Elect': 'Elect person entity position time place', 
'Convict': 'Convict defendant adjudicator crime time place', 
'Sentence': 'Sentence defendant adjudicator crime sentence time place', 
'Charge_Indict': 'charge indict defendant prosecutor adjudicator crime time place', 
'Release_Parole': 'Release parole person entity crime time place',
'Appeal': 'Appeal defendant prosecutor adjudicator crime time place',
'Demonstrate': 'Demonstrate entity time place',
'Trial_Hearing': 'Trial hearing defendant prosecutor adjudicator crime time place',
'Divorce': 'Divorce person time place ',
'Sue': 'Sue plaintiff defendant adjudicator crime time place',
'Transfer_Ownership': 'Transfer ownership buyer seller beneficiary artifact price time place',
'Fine': 'Fine entity adjudicator money crime time place',
'Acquit': 'Acquit defendant adjudicator crime time place',
'Execute': 'Execute person agent crime time place',
'End_Org': 'End organization time place',
'Declare_Bankruptcy': 'Declare bankruptcy organization time place', 
'Be_Born': 'Be born person time place',
'Extradite': 'Extradite agent person destination origin crime time',
'Injure': 'Injure agent victim instrument time place ',
'Pardon': 'Pardon defendant adjudicator crime time place',
'Merge_Org': 'Merge organization time place'
}

QUESTIONS_ARGUMENTS = {
"Extradite": ["Agent", "Person", "Destination", "Origin", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End"],
"Merge-Org": ["Org", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"End-Position": ["Person", "Entity", "Position", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Trial-Hearing": ["Defendant", "Prosecutor", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Arrest-Jail": ["Person", "Agent", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Sentence": ["Defendant", "Adjudicator", "Crime", "Sentence", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Start-Org": ["Agent", "Org", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Declare-Bankruptcy": ["Org", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Pardon": ["Defendant", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Phone-Write": ["Entity", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End"],
"Marry": ["Person", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Fine": ["Entity", "Adjudicator", "Money", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Start-Position": ["Person", "Entity", "Position", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Release-Parole": ["Person", "Entity", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Acquit": ["Defendant", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"End-Org": ["Org", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Nominate": ["Person", "Agent", "Position", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Transfer-Money": ["Giver", "Recipient", "Beneficiary", "Money", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Appeal": ["Defendant", "Prosecutor", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Convict": ["Defendant", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Transfer-Ownership": ["Buyer", "Seller", "Beneficiary", "Artifact", "Price", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Sue": ["Plaintiff", "Defendant", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Be-Born": ["Person", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Transport": ["Agent", "Artifact", "Vehicle", "Price", "Origin", "Destination", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End"],
"Execute": ["Person", "Agent", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Demonstrate": ["Entity", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Plce"],
"Attack": ["Attacker", "Target", "Instrument", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Elect": ["Person", "Entity", "Position", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Meet": ["Entity", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Charge-Indict": ["Defendant", "Prosecutor", "Adjudicator", "Crime", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Divorce": ["Person", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Injure": ["Agent", "Victim", "Instrument", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"],
"Die": ["Agent", "Victim", "Instrument", "Time_Within", "Time_Ending", "Time_Starting", "Time_After", "Time_At_Beginning", "Time_At_End", "Place"]
}

INVERSE_QUESTIONS = {v: k for k, v in QUESTIONS.items()}

def iob2bioes(tags):

    new_tags = []
    for i, tag in enumerate(tags):
       # print(tag)
        tag = tag.replace('S-', 'B-')
        tag = tag.replace('E-', 'I-')
        if '-' not in tag:
            if tag != 'O':
                tag = 'O' 
        if '0' in tag: print('WTF')
        if tag == 'O':
            new_tags.append(tag)
        else:
            split = tag.split('-')[0]
            if split == 'B':
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif split == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                import pdb
                pdb.set_trace()
                raise TypeError("Invalid IOB format.")
    return new_tags

RELATION_LABELS = ['Other', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
                   'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                   'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
                   'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                   'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                   'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                   'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                   'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                   'Content-Container(e1,e2)', 'Content-Container(e2,e1)']

for i, j in zip(LABELS, RELATION_LABELS[1:]):
    print(i, j)

TRIGGER_LABELS = []

#NOT_IN_TRAIN = ["Marry", "Trial_Hearing", "Arrest_Jail", "Acquit", 'Attack', 'Declare_Bankruptcy']
#IN_TRAIN = ['Attack', 'Transport', 'Die', 'Meet', 'Sentence','Arrest_Jail', 'Transfer_Money', 'Elect','Transfer_Ownership', 'End_Position']
#IN_TRAIN = ['Attack', 'Transport', 'Die', 'Meet', 'Sentence']
#IN_TRAIN = ['Attack', 'Transport', 'Die']
#IN_TRAIN = ['Attack']

all_events = 0
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append((pattern, i))
    return matches

idx_semeval = 8000
event_idx = 0

ENTITY_LABELS = ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']

#import spacy
#from spacy.tokenizer import Tokenizer
#nlp = spacy.load("en_core_web_sm")
#nlp.tokenizer = Tokenizer(nlp.vocab)

all_entities, upper_entities = 0, 0
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
#        if 'train' in fname:
#            continue
        if 'json' not in fname:
            continue
        if 'semeval' in fname:
            continue
        print('\t%s' % os.path.join(dirName, fname))

        with open(os.path.join(dirName, fname), "r") as read_file:
            data = json.load(read_file)
        
        print(len(data))
        
        json_data = {"data": [{"title": "Super_Bowl_50", "paragraphs": []}]}
        
        FOLDER = '/home/eboros/projects/DATA/ace_2005_td_v7_LDC2006T06/ace_2005_td_v7/data/English/timex2norm/'
        from bs4 import BeautifulSoup

        with open(os.path.join(dirName, fname.replace('.json', '.tsv')), 'w') as f:
#            f.write('TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC\n')
            with open(os.path.join(dirName, fname.replace('.json', '_semeval.json')), 'w') as f_mrc:
                with open(os.path.join(dirName, fname.replace('.json', '_semeval.txt')), 'w') as f_out:
                    with open(os.path.join(dirName, fname.replace('.json', '_tags.txt')), 'w') as f_tags:
                        with open(os.path.join(dirName, fname.replace('.json', '_text.txt')), 'w') as f_text:

                            max_event = []
                            
                            doc_id = None
                            for sentence in data:
                                with open(os.path.join(FOLDER, sentence['event_id'] + '.apf.xml'), 'r') as xml:
                                    soup = BeautifulSoup(xml, features="lxml")
                                
                                if sentence['event_id'].split('.')[0] != doc_id:
                                    
                                    doc_id = sentence['event_id'].split('.')[0]
                                    print('DOC ID', doc_id)
                                    f.write('-DOCSTART-	-' + str(doc_id) + '-	O	O\n\n')
                                
                                
#                                all_entities, all_entity_tags = [], []
#                                for entity_mention in soup.find_all('entity'):
#                                    entities = [entity.extent.text.strip().replace('\n', ' ').replace('\t', ' ') for entity in entity_mention.find_all('entity_mention')]
#                                    entities = [entity for entity in entities if len(entity) > 1 and (entity not in all_entities)]
#                                    entities = list(set(entities))
##                                    import pdb;pdb.set_trace()
#                                    all_entity_tags += [entity_mention.get('type') for entity in entities]
#                                    all_entities += entities
                                
                                
#                                print(all_entities)
                                words = sentence['words']
                                trigger_labels = ['O'] * len(words)
                                argument_labels1 = ['O'] * len(words)
                                argument_labels2 = ['O'] * len(words)
                                
                                overall_arguments = []
                                
                                entity_labels = ['O'] * len(words)
                                event_id = sentence['event_id']
                                positions = ['(0,0)'] * len(words)
                                trigger_texts = []
                                count_triggers = []
                                
#                                sentiment = classifier(str(clean_str(' '.join(words))))
#                                score = sentiment[0]['score']
#                                sentiment = sentiment[0]['label'].lower()
                                
                                if 'golden-event-mentions' in sentence:
                                    for idx_trigger, golden_event_mention in enumerate(sentence['golden-event-mentions']):
                                        count_triggers.append(golden_event_mention['trigger']['text'])
                                        
                                
                                entity_heads = dict()
                                if 'golden-entity-mentions' in sentence:
                                    for idx_trigger, golden_event_mention in enumerate(sentence['golden-entity-mentions']):
                                        text_entity = golden_event_mention["text"]
                                        head_entity = golden_event_mention["head"]
                                        entity_heads[text_entity] = head_entity['text']

                                if 'golden-event-mentions' in sentence:
                                    

#                                    print(sentiment, score, text_trigger, trigger_label)

                                    if len(count_triggers) > 0:
                                 

                                        json_line = \
                                            {
                                                "context": clean_str(' '.join(words)),# + ' [SEP] ',
                                                "qas": [
                                                ],
                                            }
                                        for key, question in zip(QUESTIONS.keys(), QUESTIONS.values()):
                                            
#                                            NOT_IN_TRAIN = LABELS[:2]
#                                            if 'train' in fname:
#                                                if key not in IN_TRAIN:
#                                                    continue
#                                            if 'test' in fname:
#                                                
#                                                if key in IN_TRAIN:
#                                                    continue

                                            json_line['qas'].append({
                                                        "id": str(event_idx),
                                                        "is_impossible": True,
                                                        "question": question,
                                                        "answers": [],
                                                    })
                                            event_idx += 1


                                    for idx_trigger, golden_event_mention in enumerate(sentence['golden-event-mentions']):
                                        words_final = words.copy()
                                        start_trigger = golden_event_mention['trigger']['start']
                                        end_trigger = golden_event_mention['trigger']['end']
                                        start_position_trigger = golden_event_mention['trigger']['start_position']
                                        end_position_trigger = golden_event_mention['trigger']['end_position']
                                        trigger_label = golden_event_mention['event_type'].split(':')[-1].replace('-', '_')
                                        
            #                            print(golden_event_mention)
                                        trigger_texts.append(golden_event_mention['trigger']['text'])
#                                        if 'train' in fname:
#                                            if trigger_label not in IN_TRAIN:
#                                                continue
#                                        if 'test' in fname:
#                                            if trigger_label not in NOT_IN_TRAIN:
#                                                continue

#                                        if 'train' in fname:
#                                            if trigger_label in NOT_IN_TRAIN:
#                                                continue
                    #                    print(words[start_trigger:end_trigger], '---', golden_event_mention['trigger'])
                                        trigger_labels[start_trigger:end_trigger] = ['B-' + trigger_label] + ['I-' + trigger_label] * (len(words[start_trigger:end_trigger])-1)
                                        trigger_labels[start_trigger] = trigger_labels[start_trigger]
                                        positions[start_trigger:end_trigger] = ['(' + str(start_position_trigger) + ',' +
                                                 str(end_position_trigger) + ')'] * len(words[start_trigger:end_trigger])
                    #                    print(trigger_labels)
            
        #                                event_idx += 1
                                        arguments = golden_event_mention['arguments']
                                        
                                        from collections import Counter
                                        a = dict(Counter([arg['start'] for arg in arguments]))
            
                                        if any((i > 1 for i in a.values())):
                                            print('-'*30)
                                            print(arguments)
#                                            import pdb;pdb.set_trace()
                                        
                                        json_arguments = []
                                        for idx, argument in enumerate(arguments):
            
                                            start_argument = argument['start']
                                            end_argument = argument['end']
                                            argument_role =  argument['role'].replace('-', '_')
                                            entity_type = argument['entity-type'].split(':')[0].replace('-', '_')
                                            
                                            
                                            if len(overall_arguments) == 0:
                                                overall_arguments.append(['O'] * len(words))
                                            
                                            found = False
                                            for idx, list_arguments in enumerate(overall_arguments):
                                                
                                                if list_arguments[start_argument] not in ['O', 'I-' + argument_role, 'B-' + argument_role]:
#                                                    import pdb;pdb.set_trace()
                                                    found = True
                                            if found == True:
                                                overall_arguments.append(['O'] * len(words))
                                                overall_arguments[-1][start_argument:end_argument] = ['B-' + argument_role] + ['I-' + argument_role] * (len(words[start_argument:end_argument])-1)
                                                overall_arguments[-1][start_argument] = overall_arguments[-1][start_argument]
                
                                            else:
                                                overall_arguments[-1][start_argument:end_argument] = ['B-' + argument_role] + ['I-' + argument_role] * (len(words[start_argument:end_argument])-1)
                                                overall_arguments[-1][start_argument] = overall_arguments[-1][start_argument]
                                                
                                                entity_labels[start_argument:end_argument] = ['B-' + entity_type] + ['I-' + entity_type] * (len(words[start_argument:end_argument])-1)
                                                entity_labels[start_argument] = entity_labels[start_argument]
                                                
                                            json_arguments.append({
                                                                 "text": clean_str(' '.join(words[start_argument:end_argument])),
                                                                 "role": argument_role,
                                                                 "type": entity_type,
                                                                 "start_position": clean_str(' '.join(words)).index(clean_str(clean_str(' '.join(words[start_argument:end_argument])))),
                                                                 "end_position": clean_str(' '.join(words)).index(clean_str(clean_str(' '.join(words[start_argument:end_argument])))) + len(clean_str(' '.join(words[start_argument:end_argument]))),
                                                                 "head": clean_str(entity_heads[argument["text"]]).replace('$', '$ ')
                                                                })
                                                
                                            print(' '.join(words[start_argument:end_argument]), '--', argument['text'])

                                                    
#                                            if argument_labels1[start_argument] not in ['O', 'I-' + argument_role, 'B-' + argument_role]:
#                                                import pdb;pdb.set_trace()
#                                                argument_labels2[start_argument:end_argument] = ['B-' + argument_role] + ['I-' + argument_role] * (len(words[start_argument:end_argument])-1)
#                                                argument_labels2[start_argument] = argument_labels2[start_argument]
#            
#                                            else:
#                                                argument_labels1[start_argument:end_argument] = ['B-' + argument_role] + ['I-' + argument_role] * (len(words[start_argument:end_argument])-1)
#                                                argument_labels1[start_argument] = argument_labels1[start_argument]
#                                                
#                                                entity_labels[start_argument:end_argument] = ['B-' + entity_type] + ['I-' + entity_type] * (len(words[start_argument:end_argument])-1)
#                                                entity_labels[start_argument] = entity_labels[start_argument]


#                                        words_final[start_trigger:end_trigger] = ['<e1>', golden_event_mention['trigger']['text'], '</e1>']

                                        
            #                            f_mrc.write('"id": "'+str(event_idx)+'",\n')
            #                            f_mrc.write('"is_impossible": False,\n')
            #                            f_mrc.write('"question": "What are the triggers?",\n')
#                                        print('-----')
#                                        print(clean_str(' '.join(words)))
                                        
                                        text_trigger = clean_str(golden_event_mention['trigger']['text'])
                                        text_trigger = text_trigger.replace('re)merger', 're merger')
#                                        print(clean_str(golden_event_mention['trigger']['text']))
#                                        print('-----')
        
        
                                        answer = {"text": clean_str(golden_event_mention['trigger']['text']), "answer_start": clean_str(' '.join(words)).index(clean_str(text_trigger)),
                                                  'arguments': json_arguments}
            #                            import pdb;pdb.set_trace()
#                                        print(answer, trigger_label)
                                        
                                        question_trigger = QUESTIONS[trigger_label]
                                        for idx, question in enumerate(json_line['qas']):
                                            
                                            if question['question'] == question_trigger:
                                                
                                                                 
                                        
                                                
#                                                if 'train' in fname:
##                                                    import pdb;pdb.set_trace()
#                                                    if trigger_label not in IN_TRAIN:
#                                                        continue
#

                                                json_line['qas'][idx]['answers'].append(answer)
                                                
                                                json_line['qas'][idx]['is_impossible'] = False
                                        
                                        sent = clean_str(' '.join(words_final))
                                        sent = sent.replace('<e1', '<e1>')
                                        sent = sent.replace('< e1', '</e1>')
                                        f_out.write(trigger_label + '\t' + sent + '\n')
#                                        if trigger_label not in LABELS:
#                                            LABELS.append(trigger_label)
                                            

        #                                    print(LABELS)
        #                                if 'test' in fname:
                                            
            #                            f_mrc.write('{"text": "'+clean_str(golden_event_mention['trigger']['text']).lower()+'", "answer_start": '+str(start_trigger)+'},\n')          
            #                            print(clean_str(' '.join(words_final)) + '\t' + clean_str(golden_event_mention['trigger']['text']) + '\t' + str(start_trigger) + '\t' + str(end_trigger) + '\t' + sentiment)
            #                            f_mrc.write(str(event_idx) + '\t' + clean_str(' '.join(words)).lower() + '\t' + clean_str(golden_event_mention['trigger']['text']).lower() + '\t' + str(start_trigger) + '\t' + str(end_trigger) + '\t' + sentiment + '\n')

#                                doc = nlp(' '.join(words))
                                
#                                all_entities, all_entity_tags = [], []
#                                entity_labels = ['O'] * len(words)
#                                for ent in doc.ents:
##                                    if 'Job_Title'
#                                    print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                                    all_entity_tags.append(ent.label_)
#                                    all_entities.append(ent.text)
#
#                                for entity, entity_tag in zip(all_entities, all_entity_tags):
##                                            print('Adding', entity)
#                                    entity_tokens = entity.split(' ')
#                                    matches = subfinder(words, entity_tokens)
#                                    for match in matches:
#                                        _, idx_match = match
#                                        entity_labels[idx_match:idx_match+len(entity_tokens)] = ['B-'+entity_tag] + ['I-'+entity_tag]* (len(entity_tokens)-1)

#                                pos_tags = [pos for token, pos in pos_tag(words)]
#                    #            print(tags, tokens)
#                                
#                                conlltags = [(token, pos, tg) for token, pos, tg in zip(words, pos_tags, argument_labels1)]
#                                ne_tree = conlltags2tree(conlltags)
#                    
#                            
#                                original_text = []
#                                for subtree in ne_tree:
#                                    # skipping 'O' tags
#                                    if type(subtree) == Tree:
#                                        original_label = subtree.label()
#                                        original_string = " ".join([token for token, pos in subtree.leaves()])
#                                        original_text.append((original_string, original_label))
#                                        
###                                print(original_text)
#                                for element in original_text:
#                                    entity_tokens, entity_tag = element
#                                    entity_tokens = entity_tokens.split(' ')
#                                    matches = subfinder(words, entity_tokens)
#                                    for match in matches:
#                                        _, idx_match = match
##                                        entity_tag = 'Entity'
#                                        if '<'+entity_tag+'>' not in TRIGGER_LABELS:
#                                            TRIGGER_LABELS.append('<'+entity_tag+'>')
#                                            TRIGGER_LABELS.append('</'+entity_tag+'>')
#                                        words[idx_match:idx_match+len(entity_tokens)] = ['<'+entity_tag+'>'] + words[idx_match:idx_match+len(entity_tokens)] + ['</'+entity_tag+'>']
#                                        argument_labels1[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels1[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        argument_labels2[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels2[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        trigger_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + trigger_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        entity_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + entity_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        positions[idx_match:idx_match+len(entity_tokens)] = ['(0,0)'] + positions[idx_match:idx_match+len(entity_tokens)] + ['(0,0)']
##        
#                                conlltags = [(token, pos, tg) for token, pos, tg in zip(words, pos_tags, argument_labels2)]
#                                ne_tree = conlltags2tree(conlltags)
#                            
#                                original_text = []
#                                for subtree in ne_tree:
#                                    # skipping 'O' tags
#                                    if type(subtree) == Tree:
#                                        original_label = subtree.label()
#                                        original_string = " ".join([token for token, pos in subtree.leaves()])
#                                        original_text.append((original_string, original_label))
#                                        
##                                print(original_text)
#                                for element in original_text:
#                                    entity_tokens, entity_tag = element
#                                    entity_tokens = entity_tokens.split(' ')
#                                    matches = subfinder(words, entity_tokens)
#                                    for match in matches:
#                                        _, idx_match = match
##                                        entity_tag = 'E'
#                                        words[idx_match:idx_match+len(entity_tokens)] = ['<'+entity_tag+'>'] + words[idx_match:idx_match+len(entity_tokens)] + ['</'+entity_tag+'>']
#                                        argument_labels1[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels1[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        argument_labels2[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels2[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        trigger_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + trigger_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        entity_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + entity_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                        positions[idx_match:idx_match+len(entity_tokens)] = ['(0,0)'] + positions[idx_match:idx_match+len(entity_tokens)] + ['(0,0)']
#        
        #                        if 'Davies' in words:
        #                            import pdb;pdb.set_trace()
                                    
        #                        print(words)
#                                print(len(trigger_texts))
#                                all_events += len(trigger_texts)
        
#        
#                                for idx_entity, golden_entity_mention in enumerate(sentence['golden-entity-mentions']):
#                                    start_entity = golden_entity_mention['head']['start']
#                                    end_entity = golden_entity_mention['head']['end']
#                                    entity_text = golden_entity_mention['head']['text']
#                                    entity_type = golden_entity_mention['entity-type'].split(':')[0]
##                                    if 'British' in entity_text:
##                                        import pdb;pdb.set_trace()
#                                    if entity_type in ENTITY_LABELS:
#    #                                    print(entity_text, entity_type)
#                                        entity_labels[start_entity:end_entity] = ['B-' + entity_type] + ['I-' + entity_type] * (len(words[start_entity:end_entity])-1)
#                                        entity_labels[start_entity] = entity_labels[start_entity]
                                        
#                                        entity_text = entity_text.replace(',', ' ,')
#                                        entity_tokens = entity_text.split(' ')
#                                        matches = subfinder(words, entity_tokens)
#                                        if len(matches) > 0:
#                                            for match in [matches[0]]:
#                                                _, idx_match = match
#        
#                                                words[idx_match:idx_match+len(entity_tokens)] = ['<'+entity_type+'>'] + words[idx_match:idx_match+len(entity_tokens)] + ['</'+entity_type+'>']
#                                                argument_labels1[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels1[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                argument_labels2[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels2[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                trigger_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + trigger_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                entity_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + entity_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                positions[idx_match:idx_match+len(entity_tokens)] = ['(0,0)'] + positions[idx_match:idx_match+len(entity_tokens)] + ['(0,0)']
#                                        else:
#                                            print('----------', entity_text, '---------', ' '.join(words))

                                
                                if len(trigger_texts) == 0:
#                                    
                                        words = sentence['words']
#                                        
#                                        doc = nlp(' '.join(words))
##                                        import pdb;pdb.set_trace()
#                                        
#                                        all_entities, all_entity_tags = [], []
#                                        for ent in doc.ents:
##                                            print(ent.text, ent.start_char, ent.end_char, ent.label_)
#                                            all_entity_tags.append(ent.label_)
#                                            all_entities.append(ent.text)
                                            
#                                        import pdb;pdb.set_trace()

                                        trigger_labels = ['O'] * len(words)
                                        argument_labels1 = ['O'] * len(words)
                                        argument_labels2 = ['O'] * len(words)
                                        entity_labels = ['O'] * len(words)
                                        positions = ['(0,0)'] * len(words)
#
#                                        for entity, entity_tag in zip(all_entities, all_entity_tags):
##                                            print('Adding', entity)
#                                            entity_tokens = entity.split(' ')
#                                            matches = subfinder(words, entity_tokens)
#                                            for match in matches:
#                                                _, idx_match = match
#                                                entity_labels[idx_match:idx_match+len(entity_tokens)] = ['B-'+entity_tag] + ['I-'+entity_tag]* (len(entity_tokens)-1)
                                        

#                                        add other entities
#                                        for entity, entity_tag in zip(all_entities, all_entity_tags):
##                                            print('Adding', entity)
#                                            entity_tokens = entity.split(' ')
#                                            matches = subfinder(words, entity_tokens)
#                                            for match in matches:
#                                                _, idx_match = match
#                                                
##                                                entity_tag = 'Entity'
#                                                
#                                                if '<'+entity_tag+'>' not in TRIGGER_LABELS:
#                                                    TRIGGER_LABELS.append('<'+entity_tag+'>')
#                                                    TRIGGER_LABELS.append('</'+entity_tag+'>')
#                                                words[idx_match:idx_match+len(entity_tokens)] = ['<'+entity_tag+'>'] + words[idx_match:idx_match+len(entity_tokens)] + ['</'+entity_tag+'>']
#                                                argument_labels1[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels1[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                argument_labels2[idx_match:idx_match+len(entity_tokens)] = ['O'] + argument_labels2[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                trigger_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + trigger_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                entity_labels[idx_match:idx_match+len(entity_tokens)] = ['O'] + entity_labels[idx_match:idx_match+len(entity_tokens)] + ['O']
#                                                positions[idx_match:idx_match+len(entity_tokens)] = ['(0,0)'] + positions[idx_match:idx_match+len(entity_tokens)] + ['(0,0)']

                                        
#                                        sentiment = classifier(str(clean_str(' '.join(words))))
#                                        score = sentiment[0]['score']
#                                        sentiment = sentiment[0]['label'].lower()
#
#                                        print(sentiment, score)
                                        json_line = \
                                            {
                                                "context": clean_str(' '.join(words)),# + ' [SEP] ' + sentiment,
                                                "qas": [],
                                            }
                                        for key, question in zip(QUESTIONS.keys(), QUESTIONS.values()):

#                                            if 'train' in fname:
#                                                if key not in IN_TRAIN:
#                                                    continue
#                                            if 'test' in fname:
#                                                if key in IN_TRAIN:
#                                                    continue
#
#                                            
                                            
                                            json_line['qas'].append({
                                                        "id": str(event_idx),
                                                        "is_impossible": True,
                                                        "question": question,
                                                        "answers": [],
                                                    })
                                            event_idx += 1
                       
                                        start_trigger = 0
                                        end_trigger = len(clean_str(' '.join(words)).split(' '))

                                        for idx, question in enumerate(json_line['qas']):
#                                            print(question['answers'])
                                            if len(question['answers']) > 0:
                                                
                                                f_tags.write(str(question['id']) + '\t')
                                                for answer in question['answers']:
                                                    f_tags.write(str(['text']) + ', ') 
                                                f_tags.write('\t' + json_line['context'] + '\n')
#                                                print('----'+str(question['id']) + '\t' + str(question['answers'][0]['text']))

                                            else:
                                                f_tags.write(str(question['id']) + '\t' + str('NONE') +'\t' +INVERSE_QUESTIONS[question['question']].lower() + '\t' + json_line['context'] + '\n')

                                        for idx, question in enumerate(json_line['qas']):
                                            json_line['qas'][idx]['question'] = question['question'].lower()

                                        json_data['data'][0]['paragraphs'].append(json_line)

                                        event_idx += 1
                                        
                                        #fine-tune language model
                                        f_text.write(clean_str(' '.join(words)) + '\n')
                                        
                                        # Write NER
                                        for (word, trigger_label, entity_label, role1, role2, position) in zip(words, trigger_labels, entity_labels,
                                                argument_labels1, argument_labels2, positions):
                                            word = word.replace(u'\xa0', u' ')
                #                            if '-' in trigger_label:
                #                                trigger_label = trigger_label[:trigger_label.index('-')] + '-Trigger'
                                            for w in word.split(' '):
                #                                f.write(str(event_id) + '\t' + w + '\t' + trigger_label + '\t' +  entity_label + 
                #                                        '\t' + role1 + '\t' + role2 + '\t' + position + '\n')
                                                f.write(w + '\t' + trigger_label +  '\t' + position + '\n')
                                            if trigger_label != 'O':
                                                if 'B-' in trigger_label:
        #                                            all_entities += 1
                                                    if w[0].isupper():
                                                        upper_entities += 1
                                                        
                                        f.write('\n')

                                else:
#                                    import pdb;pdb.set_trace()
                                    
                                    # add markers to QA
                                    sentence = ' '.join(words)
                                    
                                    json_line['context'] = clean_str(sentence)# + ' [SEP] ' + sentiment
                                    
                                    json_line['context'] = json_line['context'].replace('<ORG> take </ORG>', 'take')
                                    json_line['context'] = json_line['context'].replace('<FAC> go <FAC> inside </FAC> </FAC>', 'go inside')
                                    json_line['context'] = json_line['context'].replace('take </Seller> over', 'take over </Seller>')
                                    json_line['context'] = json_line['context'].replace('<E> take </E> over', 'take over')
                                    json_line['context'] = json_line['context'].replace('<Entity> take </Entity> over', 'take over')
                                    json_line['context'] = json_line['context'].replace('<E> go <E> inside </E> </E>', 'go inside')
                                    json_line['context'] = json_line['context'].replace('<Entity> go <Entity> inside </Entity> </Entity>', 'go inside')
                                    
                                    json_line['context'] = json_line['context'].replace('<Convict> not </Convict> found', 'not found')
                                    json_line['context'] = json_line['context'].replace('<Destination> go <Destination> inside', '<Destination> go inside <Destination>')
#                                    if 'take ' in json_line['context']:
#                                        import pdb;pdb.set_trace()
                                        
#                                    print(json_line['context'])
                                    
                                    for question in json_line['qas']: 
                                        if len(question['answers']) > 0:
                                            for idx_answer, answer in enumerate(question['answers']):
                                                text_answer = answer['text']
                                                text_answer = text_answer.replace('charges of widespread child sexual abuse in', 'charges')
                                                text_answer = text_answer.replace('re)merger', 're merger')
                                                if text_answer == 'lost t':
                                                    text_answer = 'lost'
                                                print(sentence, '-----', text_answer)
                                                question['answers'][idx_answer]['answer_start'] = json_line['context'].index(text_answer)
                                    # add markers to QA
                                    for idx, question in enumerate(json_line['qas']):
                                        if len(question['answers']) > 0:
                                                f_tags.write(str(question['id']) + '\t')
                                                for answer in question['answers']:
                                                    f_tags.write(str(answer['text']) + ', ') 
                                                f_tags.write('\t' + INVERSE_QUESTIONS[question['question']].lower() + '\t' + json_line['context'] + '\n')
                                        else:
                                            f_tags.write(str(question['id']) + '\t' + str('NONE') + '\t' + INVERSE_QUESTIONS[question['question']].lower() + '\t' + json_line['context'] + '\n')
#                                            json_line['qas'][idx]['answers'].append(answer)

                                    f_text.write(clean_str(' '.join(words)) + '\n')

                                    for idx, question in enumerate(json_line['qas']):
                                        json_line['qas'][idx]['question'] = question['question'].lower()


                                    json_data['data'][0]['paragraphs'].append(json_line)
                             

                                    # Write NER
                                    assert len(trigger_labels) == len(words)
                                    assert len(trigger_labels) == len(entity_labels)
#                                    assert len(trigger_labels) == len(argument_labels2)
#                                    assert len(trigger_labels) == len(argument_labels1)
                                    assert len(trigger_labels) == len(positions)
                                    
                                    import numpy as np
#                                    for (word, trigger_label, entity_label, role1, role2, position) in zip(words, trigger_labels, entity_labels,
#                                            argument_labels1, argument_labels2, positions):
                                    if len(overall_arguments) > 1:
                                        stack = np.stack(overall_arguments, 1)
#                                                import pdb;pdb.set_trace()
                                    elif len(overall_arguments) == 1:
                                        stack = overall_arguments[0]
                                    else:
                                        stack = ['O'] * len(words)
                                        
#                                    import pdb;pdb.set_trace()

                                    for idx, item in enumerate(zip(words, trigger_labels, entity_labels, positions)):
                                        word, trigger_label, entity_label, position = item
                                        word = word.replace(u'\xa0', u' ')
            #                            if '-' in trigger_label:
            #                                trigger_label = trigger_label[:trigger_label.index('-')] + '-Trigger'
                                        for w in word.split(' '):
            #                                f.write(str(event_id) + '\t' + w + '\t' + trigger_label + '\t' +  entity_label + 
            #                                        '\t' + role1 + '\t' + role2 + '\t' + position + '\n')
                                            if len(overall_arguments) <= 1:
#                                                print(stack[idx])
                                                f.write(w + '\t' + trigger_label +  '\t' +  entity_label + 
                                                       '\t' + stack[idx] + '\t' + position + '\n')
                                            else:
                                                f.write(w + '\t' + trigger_label +  '\t' +  entity_label + 
                                                       '\t' + '\t'.join(stack[idx]) + '\t' + position + '\n')

                                        if trigger_label != 'O':
                                            if 'B-' in trigger_label:
    #                                            all_entities += 1
                                                if w[0].isupper():
                                                    upper_entities += 1
                                    f.write('\n')

                                    
                                    
                                    
                            json.dump(json_data, f_mrc, indent=1)

                          
print(all_events)
                
print(all_entities, upper_entities)
                
print(TRIGGER_LABELS)
                
                
                
                
                
