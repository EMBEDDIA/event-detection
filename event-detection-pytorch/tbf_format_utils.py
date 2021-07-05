
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)  # for reproducibility

"""
-- file          tbf_format_utils.py
-- author        Olivier Ferret
-- version       1.0
-- created       may 8, 2018
-- last_revised  may 8, 2018

various classes and functions for dealing with the tbf format
"""

# for Python3 compatibility
#from builtins import *


class EventMention:
    """
    event mention in a document
    """
    def __init__(self, id, start, end, text, type):
        """
        initialize a new event mention with its core data
        """
        self.id = id
        self.start_offset = start
        self.end_offset = end
        self.text = text
        self.type = type

    def print(self, fhd):
        """
        print the data of the current event mention
        """
        print(self.id, '\t', self.start_offset, '\t', self.end_offset, '\t', self.text,
              '\t', self.type, file=fhd, sep='')
        
    def printCore(self, fhd):
        """
        print the core data of the current event mention
        """
        
        print(self.start_offset, '\t', self.end_offset, '\t', self.text,
              '\t', self.type,'\t', self.id, file=fhd, sep='', end='')        

    def overlap(self, em):
        """
        is self included in em?
        """
        #return not (self.start_offset > em.end_offset or self.end_offset < em.start_offset)
        return self.start_offset >= em.start_offset and self.end_offset <= em.end_offset
    
    def equal(self, em):
        """
        is self fully equal to em?
        """
        return self.start_offset == em.start_offset and self.end_offset == em.end_offset \
               and self.type == em.type and self.text == em.text
           
    def equal_with_inclusion(self, em):
        """
        is self included in em or em included in self?
        """
        return (self.overlap(em) or em.overlap(self)) and self.type == em.type and \
            (self.text in em.text or em.text in self.text)


class DocumentEventAnnotations:
    """
    annotations about events for a document
    """
    def __init__(self, id):
        """
        initialization of a new instance with an identifier
        """
        self.id = id
        self.eventMentions = []

    def print(self, fhd):
        """
        print the set of event mentions of the document
        """
        print(self.id, file=fhd)
        for em in self.eventMentions:
            em.print(fhd)

def createEventAnnotations(results):
    """
    read the annotations in tbf format from a file for several documents
    docId, id_event, start, end, text, event_type
    """
    
#    for idx, item in 
    docs = {}
    for result in results:
        docId, id_event, start, end, text, event_type = result
        if docId not in docs:
            docs[docId] = DocumentEventAnnotations(docId)
        mention = EventMention(id_event, start, end, text, event_type.lower())
        docs[docId].eventMentions.append(mention)
        
#    docs = {}
#    for fl in fhd:
#        fl = fl.rstrip()
#        if fl[0] == '#':
#            fields = fl.split()
#        elif fl != '':
#            fields = fl.split('\t')
#        if fields[0] == '#BeginOfDocument':
#            docId = fields[1]
#            docAnnots = DocumentEventAnnotations(docId)
#        elif fields[0] == '#EndOfDocument':
#            docs[docId] = docAnnots
#        else:
#            start, end = fields[3].split(',')
#            mention = EventMention(fields[2], int(start), int(end), fields[4], fields[5].lower())
#            docAnnots.eventMentions.append(mention)
    return docs
  
def loadEventAnnotations(fhd):
    """
    read the annotations in tbf format from a file for several documents
    """
    docs = {}
    for fl in fhd:
        fl = fl.rstrip()
        if fl[0] == '#':
            fields = fl.split()
        elif fl != '':
            fields = fl.split('\t')
        if fields[0] == '#BeginOfDocument':
            docId = fields[1]
            docAnnots = DocumentEventAnnotations(docId)
        elif fields[0] == '#EndOfDocument':
            docs[docId] = docAnnots
        else:
            start, end = fields[3].split(',')
            mention = EventMention(fields[2], int(start), int(end), fields[4], fields[5].lower())
            docAnnots.eventMentions.append(mention)
    return docs

