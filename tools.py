import codecs
import re

def get_args():
  from optparse import OptionParser
  parser = OptionParser()
  # Options specific to classif
  parser.add_option("-c", "--corpus", dest="corpus",
                  help="JSON file for the corpus to process", metavar="CORPUS")
  parser.add_option("-d", "--document_path", dest="document_path", default="docs/id/DemamKongo",
                  help="Document to process")
  parser.add_option("-e", "--evaluate", dest="evaluate", 
                  default=False, action="store_true",      
                  help = "Perform Evaluation")
  parser.add_option("-i", "--is_clean", dest="is_clean",
                    action = "store_true", default=False, help="If activated, no boilerplate removal will be applied (e.g. the document will processed as it is)")
  parser.add_option("-l", "--language", dest="language", default ="id",
                  help="Language to process (ISO 639 2 letters)")
  parser.add_option("-o", "--out", dest="name_out",
                    default = "test.out", help="Name of out file")
  parser.add_option("-r", "--ratio", dest="ratio", 
                  default =0.8, type="float", 
                  help="Defines the threshold for the relative size of the substrings (e.g. 0.8 meaning that substrings sharing 80% of the Named Entity will be considered.")
  parser.add_option("-v", "--verbose",
                   action="store_true", dest="verbose", default=False,
                   help="Show status messages to stdout")
  parser.add_option("-s", "--showrelevant",
                   action="store_true", dest="showrelevant", default=False,
                   help="Show informations on files classified as relevant")
  parser.add_option("-D", "--debug",
                   action="store_true", dest="debug", default=False,
                   help="print debug information")
  (options, args) = parser.parse_args()
  return options

def effectif_from_list(liste):
  dic = {}
  for elem in liste:
    dic.setdefault(elem, 0)
    dic[elem]+=1
  return dic

def open_utf8(path,l=False):
  f = codecs.open(path,'r','utf-8')
  if  l==True:
    out = f.readlines()
    out = [re.sub("\n|\r","",x) for x in out]
  else:
    out = f.read()
  f.close()
  return out

def write_utf8(path,out):
  w = codecs.open(path,'w','utf-8')
  w.write(out)
  w.close()
