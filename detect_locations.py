#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
import os
import re
import glob
sys.path.append('./rstr_max')
from tools_karkkainen_sanders import *
from rstr_max import *
import os
from tools import *

def exploit_rstr(r,rstr, s_id_txt):
  desc = []
  for (offset_end, nb), (l, start_plage) in r.iteritems():
    ss = rstr.global_suffix[offset_end-l:offset_end]
    s_occur = set()
    for o in range(start_plage, start_plage+nb) :
      id_str = rstr.idxString[rstr.res[o]]
      s_occur.add(id_str)
    inter = s_occur.intersection(s_id_txt)
    if len(inter)>1 and len(s_occur)>len(inter):
      NE_ids = [x-len(s_id_txt) for x in s_occur.difference(s_id_txt)]
      l_distances = []
      for d in inter:
        l_distances.append(min(d, len(s_id_txt)-d-1))
      desc.append([ss, NE_ids, sorted(l_distances)])
  return desc

def get_score(ratio, dist):
  score = pow(ratio, 1+dist[0]*dist[1])
  return score

def filter_desc(desc, l_rsc, s_id_txt, loc=False):
  out = []
  for ss, dis_list, distances in desc:
    for id_dis in dis_list:
      disease_name = l_rsc[id_dis]
      ratio = float(len(ss))/len(disease_name)
      if ss[0]!=disease_name[0]:
        if loc==True:
          continue#for country names the first character should not change
        else:
          ratio = ratio-0.1#penalty
      score = get_score(ratio, distances)
      out.append([score, disease_name, ss, distances])
  return sorted(out,reverse=True)

def get_desc(string, rsc, loc = False):
  s_id_txt = set()
  rstr = Rstr_max()
  cpt = 0
  l_rsc = rsc.keys()
  for s in string:
    rstr.add_str(s)
    s_id_txt.add(cpt)
    cpt+=1
  for r in l_rsc:
    rstr.add_str(r)
  r = rstr.go()
  desc = exploit_rstr(r,rstr, s_id_txt)
  res = filter_desc(desc, l_rsc, s_id_txt, loc)
  return res 

def zoning(string):
  z = re.split("<p>", string)
  z = [x for x in z if x!=""]
  return z

def analyze(string, ressource, options): 
  zones = zoning(string)
  dis_infos = get_desc(zones, ressource["diseases"])
  events = []
  loc_infos = []
  if len(dis_infos)>0:
    loc_infos = get_desc(zones, ressource["locations"], True)
    if len(loc_infos)==0:
      loc = [ressource["locations"]["default_value"]]
    else:
      loc = [loc_infos[0][1]]
    town_infos = get_desc(zones, ressource["towns"], True)
    if len(town_infos)>0:
      for t in town_infos:
        if t[0]<options.ratio:break
        loc.append((t[1], t[0]))
    for dis in dis_infos[:1]:
      events.append([dis[1], loc])
  dic_out = {"events":events, "dis_infos":dis_infos, "loc_infos":loc_infos}
  return dic_out

def get_towns(path):
  liste = eval(open_utf8(path))
  dic = {}
  for town, pop, region in liste:
    dic[town] = [pop, region]
  return dic

def get_ressource(lg):
  dic = {}
  for rsc_type in ["diseases", "locations"]:
    try:
      path = "ressources/%s_%s.json"%(rsc_type, lg)
      dic[rsc_type] = eval(open_utf8(path))
    except Exception as e:
      print e
      print "Ressource '%s' not found\n ->exiting"%path
      exit()
  try:
    path_towns= "ressources/towns_%s.json"%lg
    dic["towns"] = get_towns(path_towns)
  except:
    print "Ressource '%s' not found"%path_towns
    dic["towns"]={}
  return dic

def open_utf8(path):
  f = codecs.open(path,"r", "utf-8")
  string = f.read()
  f.close()
  return string

def get_clean_html(path, language, isnot_clean):
  if isnot_clean == False:
    return open_utf8(path)
  try:
    tmp = "tmp/out"
    os.system("rm %s"%tmp)
    cmd = "python -m justext -s %s %s >tmp/out"%(language, path)
    os.system(cmd)
    out = open_utf8(tmp)
  except:#to improve
    print "Justext is missing"
    out = open_utf8(path)
  return out
  
def process(o):
  string = get_clean_html(o.document_path, o.language, o.isnot_clean)
  ressource = get_ressource(o.language)
  results = analyze(string, ressource, o)
  return results

if __name__=="__main__":
  options = get_args()
  try: os.makedirs("tmp")
  except: pass
  results = process(options)
  ratio = float(options.ratio)
  descriptions = eval(open("ressources/descriptions.json").read())
  for key, val in results.iteritems():
    if val[0][0]<ratio:break
    print descriptions[key]
    for v in val:
      if v[0]<ratio:break
      print "  %s"%v
