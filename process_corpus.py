from daniel import process
import glob
import sys
import json
import os
import time
import codecs
from tools import *
from daniel import get_ressource, process_results

# TODO: write test for these functions

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def open_utf8(path):
    with codecs.open(path, "r", "utf-8") as f:
        chaine = f.read()
    return chaine

# write result
def write_output(output_dic, corpus_path):
    output_path = "%s.results"%corpus_path
    output_json = json.dumps(output_dic, sort_keys=True, indent=2)
    with open(output_path, "w") as wfi:
        wfi.write(output_json)
    return output_path

def prepare_infos(infos, options):
    attr = ["is_clean","ratio","verbose","debug","name_out","showrelevant"]
    for name in attr:
        infos[name] = getattr(options, name)
    return infos 

# output list of missing docs
def list_docs_not_found(missing_docs): 
    if len(missing_docs) > 0:
        path = "tmp/files_not_found"
        print ("--\n %s files not found\n"%str(len(missing_docs)))
        print ("list here: %s\n--"%(path))
        write_utf8(path, "\n".join(missing_docs))

# look for doc - if missing, put in list to print later
def look_for_doc(doc_path, missing_doc):
    if not os.path.exists(doc_path):
        missing_doc.append(doc_path)
        return False
    return True

# return absolute path to document if abs_path exists, otherwise doc_path remain
def check_abs_path(doc_path, corpus_path):
    if os.path.abspath(corpus_path):
        doc_path = os.path.dirname(os.path.abspath(corpus_path)) + "/" + doc_path
    return doc_path

def info_process(infos, resources):
    return process(Struct(**infos), resources[infos["language"]])
    
def verbose_result(infos, results):
    if Struct(**infos).verbose or Struct(**infos).showrelevant:
        process_results(results, Struct(**infos))

def start_detection(options):
    corpus_to_process = json.load(open(options.corpus))
    cpt_proc, cpt_rel = 0, 0
    output_dic, resources = {}, {}
    missing_docs = []
  
    print ("\n Processing %s documents\n"%str(len(corpus_to_process)))
  
    for id_file, infos in corpus_to_process.iteritems(): 
        infos["document_path"] = check_abs_path(infos["document_path"], options.corpus)

        if not look_for_doc(infos["document_path"], missing_docs):
            continue
    
        cpt_proc += 1

        output_dic[id_file] = infos
    
        if "annotations" in output_dic[id_file]:
            del output_dic[id_file]["annotations"]# for evaluation
    
        infos = prepare_infos(infos, options)

        if options.verbose:
            print (infos)

        if infos["language"]  not in resources:
            resources[infos["language"]] = get_ressource(infos["language"], options)
    
        results = info_process(infos, resources)
        verbose_result(infos, results)

        if "dis_infos" in results:
            cpt_rel += 1
    
        output_dic[id_file]["annotations"] = results["events"]
        output_dic[id_file]["is_clean"] = str(output_dic[id_file]["is_clean"])

        if cpt_proc%100 == 0:
            print ("%s documents processed, %s relevant"%(str(cpt_proc), str(cpt_rel)))
    
        output_path = write_output(output_dic, options.corpus)

    list_docs_not_found(missing_docs) 

    return cpt_proc, cpt_rel, output_path

if __name__=="__main__":
    start = time.clock()
    options = get_args()
    print (options)
    if options.corpus==None:
        print ("Please specify a Json file (-c), see README.txt for more informations about the format. To use the default example :\n -c docs/Indonesian_GL.json")
        exit()
    else:
        options.document_path ="None"
    try:
        os.makedirs("tmp")
    except:
        pass
    cpt_doc, cpt_rel, output_path = start_detection(options)
    end = time.clock()
    print ("%s docs proc. in %s seconds"%(str(cpt_doc), str(round(end-start, 4))))
    print ("  %s relevant documents"%(str(cpt_rel)))
    print ("  Results written in %s"%output_path)
    if options.evaluate:
        print ("\nEvaluation\n :")
        cmd = "python evaluate.py %s %s"%(options.corpus, output_path)
        print ("-->",cmd)
        os.system(cmd)
