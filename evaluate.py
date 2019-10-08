import re
import sys, os
import json
from collections import Counter

def get_dic(path):
  f = open(path)
  dic = json.load(f)
  f.close()
  return dic

def get_verdict(GT, EV):
  if GT[0][0]!="N":
    if EV[0][0]!="N":
      verdict = "TP"
    else:
      verdict = "FN"
  else:
    if EV[0][0]!="N":
      verdict = "FP"
    else:
      verdict = "TN"
  return verdict

def store_errors(errors, infos, verdict,annot_GT, annot_eval):
  lg = infos["language"]
  if verdict=="FP":
    errors[verdict].setdefault(lg, [])
    errors[verdict][lg].append(annot_eval)
  elif verdict=="FN":
    errors[verdict].setdefault(lg, [])
    errors[verdict][lg].append(annot_GT)      
#      os.system("gedit %s"%infos["document_path"])
#      dd = input("Next ?")
  return errors

def show_errors(errors):
  for typ, dic in errors.items():
    print("****\n-%s-\n****"%typ)
    for lg, list_errors in dic.items():
      err = Counter(["--".join(x[:2]) for x in list_errors])
      print("-> %s"%lg, err.most_common(5))

def get_measures(dic, beta=1):
  TP, FP, FN = dic["TP"], dic["FP"] , dic["FN"]
  if TP==0:
    return {"Recall":0, "Prec.":0, "F%s-meas."%str(beta):0}
  R = float(TP)/(TP+FN)
  P = float(TP)/(TP+FP)
  B = beta*beta
  F = (1+B)*P*R/(B*P+R)
  return {"Recall":round(R,4), "Prec.":round(P,4), "F%s-meas."%str(beta):round(F,4)}

def get_results(dic_GT, dic_eval):
  dic_results = {x:0 for x in ["TP","FP","FN","TN"]}
  dic_lg ={}
  dic_results["Missing_GT"] = []
  errors = {"FP":{}, "FN":{}}
  for id_doc, infos in dic_eval.items():
    lg = infos["language"]
    dic_lg.setdefault(lg,{x:0 for x in ["TP","FP","FN","TN"]})
    try:annot_eval = infos["annotations"][0]
    except:continue
    if id_doc in dic_GT:
      annot_GT = dic_GT[id_doc]["annotations"][0]
      verdict = get_verdict(annot_GT, annot_eval)#TODO: add events
      dic_results[verdict]+=1
      dic_lg[lg][verdict]+=1
    else:
      dic_results["Missing_GT"].append(id_doc)
    errors = store_errors(errors, infos, verdict,annot_GT, annot_eval)
  show_errors(errors)
  if dic_results["TP"]+dic_results["FN"]==0:
    print("  No relevant documents in this Ground Truth")
  print("-"*20)
  print(dic_results)
  print("  %s annotations missing"%str(len(dic_results["Missing_GT"])))
  global_res = get_measures(dic_results)
  meas_names = global_res.keys()
  measures = [str(global_res[x]) for x in meas_names]
  print("\t"+"\t".join(meas_names))
  print("all\t"+"\t".join(measures))
  for lg , infos in dic_lg.items():
    measures = [str(get_measures(infos)[x]) for x in meas_names]
    print("%s\t"%lg + "\t".join(measures))

if len(sys.argv)!=3:
  print("USAGE : arg1=groundtruth file arg2 = result file")
  exit()

groundtruth_path = sys.argv[1]
eval_path = sys.argv[2]

dic_GT = get_dic(groundtruth_path)
dic_eval = get_dic(eval_path)

get_results(dic_GT, dic_eval)
