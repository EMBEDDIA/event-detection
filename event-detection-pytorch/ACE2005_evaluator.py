#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
np.random.seed(42)  # for reproducibility
"""
-- file          ace-event-detection-evaluator.py
-- author        Olivier Ferret
-- version       1.0
-- created       mar 6, 2018
-- last_revised  may 8, 2018

evaluation of event detection according to (li et al., 2013). Input files are in tbf format.
"""

# for Python3 compatibility
#from __future__ import (absolute_import, division,
#                        print_function, unicode_literals)
#from builtins import *

from tbf_format_utils import loadEventAnnotations, createEventAnnotations

import sys
import argparse




class EvaluationStatistics:
    """
    store statistics for computing evaluation measures
    """
    def __init__(self):
        """
        initialize a new void instance
        """
        self.tp = 0
        self.syst_em_nb = 0
        self.gold_em_nb = 0

    def recall(self):
        """
        compute recall
        """
        try:
            return float(self.tp) / float(self.gold_em_nb)
        except:
            return 0.0
        
    def precision(self):
        """
        compute precision
        """
        try:
            return float(self.tp) / float(self.syst_em_nb)
        except:
            return 0.0
        
    def f1(self):
        """
        compute f1-measure
        """
        precision = self.precision()
        recall = self.recall()
        try:
            return 2 * recall * precision / (recall + precision)
        except:
            return 0.0
        
    def print(self, fhd):
        """
        print internal statistics
        """
        print('tp\t', self.tp, '\tgold_em_nb\t', self.gold_em_nb, '\tsyst_em_nb\t',
              self.syst_em_nb, file=fhd, sep='')


def evaluationForDocument(gold, syst, stats):
    """
    match system and gold event mentions
    """
    stats.syst_em_nb += len(syst.eventMentions)
    stats.gold_em_nb += len(gold.eventMentions)
    gold_evt_mentions = list(gold.eventMentions)
    for syst_em in syst.eventMentions:        
        for idx, gold_em in enumerate(gold_evt_mentions):
            if syst_em.overlap(gold_em):
                if syst_em.type == gold_em.type:
                    stats.tp += 1
                    del gold_evt_mentions[idx]  # for avoiding using a gold mention several times
                    break
#                else:
#                    print(syst_em.type, gold_em.type, file=sys.stderr)

def load_gold(file_gold):
    f_gold = open(file_gold)
    gold_annots = loadEventAnnotations(f_gold)
    f_gold.close()
    return gold_annots

def evaluate(gold_annots, file_system):
    
    if type(file_system) == list:
         syst_annots = createEventAnnotations(file_system)
    else:
        f_system = open(file_system)
        syst_annots = loadEventAnnotations(f_system)
        f_system.close()

    if type(gold_annots) == str:
        gold_annots = load_gold(gold_annots)
    evalStats = EvaluationStatistics()
    for docId, systDocAnnots in syst_annots.items():
        goldDocAnnots = gold_annots.get(docId)
        if goldDocAnnots is None:  # document without annotation
            print('-- no gold annotation for document:', docId, file=sys.stderr)
            continue
        evaluationForDocument(goldDocAnnots, systDocAnnots, evalStats)
        
    # -- dump results
    print('precision\t', evalStats.precision(), '\trecall\t', evalStats.recall(),
          '\tf1\t', evalStats.f1(), file=sys.stdout, sep='')
    evalStats.print(sys.stderr)
    return evalStats.precision()*100.0, evalStats.recall()*100.0, evalStats.f1()*100.0
    
def main():
    # -- process arguments
    parser = argparse.ArgumentParser(description="evaluation of event detection")
    parser.add_argument("gold", type=argparse.FileType('r'),
                        help="gold standard in tbf format")
    parser.add_argument("system", type=argparse.FileType('r'),
                        help="system output to evaluate in tbf format")
    args = parser.parse_args()

    # -- load annotations
    gold_annots = loadEventAnnotations(args.gold)
#    for doc in gold_annots.itervalues():
#        doc.print(sys.stdout)
    syst_annots = loadEventAnnotations(args.system)

    # -- evaluation (mimics li's code)
    evalStats = EvaluationStatistics()
    for docId, systDocAnnots in syst_annots.items():
        goldDocAnnots = gold_annots.get(docId)
        if goldDocAnnots is None:  # document without annotation
            print('-- no gold annotation for document:', docId, file=sys.stderr)
            continue
        evaluationForDocument(goldDocAnnots, systDocAnnots, evalStats)
    # -- dump results
    print('precision\t', evalStats.precision(), '\trecall\t', evalStats.recall(),
          '\tf1\t', evalStats.f1(), file=sys.stdout, sep='')
    evalStats.print(sys.stderr)


if __name__ == '__main__':
    main()


