#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys
import os
import re
import glob
import math
sys.path.append('./rstr_max')
from tools_karkkainen_sanders import *
from rstr_max import *
import os
from tools import *
import json
import time

# ???
def get_normalized_pos(ss, text):
    l_dist = [m.start() for m in re.finditer(re.escape(ss), text)]
    l_dist = [min(x, len(text)-x) for x in l_dist]
    return [float(x)/len(text) for x in l_dist]
# ???

def exploit_rstr(r,rstr, set_id_text):
    desc = []
    # if paragraph structure is weak
    weak_struct = len(set_id_text) == 1

    for (offset_end, nb), (l, start_plage) in r.items():
        ss = rstr.global_suffix[offset_end-l:offset_end]
        s_occur = set()

        for o in range(start_plage, start_plage+nb) :
            s_occur.add(rstr.idxString[rstr.res[o]])

            inter = s_occur.intersection(set_id_text)
            
            # what is has inter???
            has_inter = (len(inter) > 1 and len(s_occur) > len(inter))

            weak_and_repeat = (weak_struct and 0 in s_occur) #needs to be in 1st paragraph

        # ????
        if has_inter or weak_and_repeat: 
            NE_ids=[x - len(set_id_text) for x in s_occur.difference(set_id_text)]
            
            if len(inter) > 1:
                l_dist = [min(pos, len(set_id_text) - pos - 1) for pos in inter]
            else:
                l_dist = get_normalized_pos(ss, rstr.global_suffix)
            
            l_dist = [round(x, 5) for x in l_dist]
            desc.append([ss, NE_ids, sorted(l_dist)])
        # ????

    return desc

def get_score(ratio, dist):
    if ratio == 1:
        ratio = 0.99
    if dist[0] == 1:
        return ratio
    elif dist[1] <= 1:
        return pow(ratio, 1+dist[0]*dist[1])
    else:
        return pow(ratio, 1+dist[0]*math.log(dist[1]))

def filter_desc(desc, l_rsc, loc=False):
    out = []
    for ss, dis_list, distances in desc:
        for id_dis in dis_list:
            entity_name = l_rsc[id_dis]
            ratio = float(len(ss))/len(entity_name)

            if ss[0].lower() != entity_name[0].lower():
                if loc:
                    #for country names the first character should not change
                    ratio = max(0, ratio-0.2)#penalty
                else:
                    if len(entity_name) < 6 and ratio < 1:
                        ratio = max(0, ratio-0.1)#penalty

            score = get_score(ratio, distances)
            out.append([score, entity_name, ss, distances])

    return sorted(out,reverse=True)

def get_desc(string, rsc, loc = False):
    set_id_text = set()
    rstr = Rstr_max()
    cpt = 0
    l_rsc = list(rsc.keys())

    for s in string:
        rstr.add_str(s)
        set_id_text.add(cpt)
        cpt+=1

    for r in l_rsc:
        rstr.add_str(r)

    r = rstr.go() # ???? should name different perhap
    desc = exploit_rstr(r,rstr, set_id_text)

    return filter_desc(desc, l_rsc, loc)

def zoning(string, options):
    z = re.split("<p>", string)

    if len(z)==1:
        z = re.split("\n", string)

    z = [x for x in z if x!=""]

    if len(z)<3:#insufficient paragraph/linebreaks structure
        sentences = re.split("\. |\</p>", string)
        sentences = [x for x in sentences if len(x)>2]

        if len(sentences)<5:#very short article
            z = [string]
        elif len(z)==2:#Title may have been extracted
            part = int(len(z[1])/2)
            z = [z[0], z[1][:part], z[1][part:]]
        else:#No usable structure
            part = int(len(string)/3)
            z = [string[:part], string[part:part*2], string[part*2:]] 

    if options.debug:
        for zone in z:
            print (re.sub("\n", "--",zone[:70]))
            print("")
        d = raw_input("Zoning ended, proceed to next step ?")

    return z

def get_implicit_location(resource, options):
    loc = resource["locations"]["default_value"]

    try:
        source = options.source
        if source in resource["sources"]:
            loc = resource["sources"][source]
    except:
        pass

    return loc

def print_top5(res):
    for elem in res[:5]:
        print ("  ",round(elem[0], 2)," \t", elem[1], "\t", elem[2])

# TODO find a better way to write this
def get_town(town_infos, loc, ratio):
    if town_infos:
        for town in town_infos:
            if town[0] < ratio:
                break
            loc.append((town[1], town[0]))

def get_event(dis_infos, loc):
    return [(dis[1],loc) for dis in dis_infos[:1] if dis_infos]

def analyze(string, resource, options): 
    zones = zoning(string, options)
    dis_infos = get_desc(zones, resource["diseases"])

    if len(dis_infos) <= 0:
        # print "No infos on diseases"
        return {"events":[], "dis_infos":dis_infos, "loc_infos":[]}

    loc_infos = get_desc(zones, resource["locations"], True)

    if options.verbose:
        print ("Top 5 name entities for diseases: ")
        print_top5(dis_infos)

        print ("Top 5 name entities for locations: ")
        print_top5(loc_infos)

    if len(loc_infos) == 0 or loc_infos[0][0] < 0.5:
        loc =  get_implicit_location(resource, options)
    else:
        loc = loc_infos[0][1]

    town_infos = get_desc(zones, resource["towns"], True)

    get_town(town_infos, loc, options.ratio)

    events = get_event(dis_infos, loc)

    return {"events":events, "dis_infos":dis_infos, "loc_infos":loc_infos}

def get_towns(path):
    liste = eval(open_utf8(path))
    dic = {}

    for town, pop, region in liste:
        dic[town] = [pop, region]

    return dic

def get_resource(lg, o):
    dic = {}
    mandatory_rsc = ["diseases", "locations"]
    for rsc_type in ["diseases", "locations", "sources"]:
        path = "resources/%s_%s.json"%(rsc_type, lg)
        if os.path.exists(path):
            try:
                dic[rsc_type] = eval(open_utf8(path))
            except Exception as e:
                if rsc_type in mandatory_rsc:
                    print ("\n  Problem with resource %s :"%path)
                    print (e)
                    exit()
                else:
                    dic[rsc_type] = {}
        else:
            if rsc_type in mandatory_rsc:
                print ("  Ressource '%s' not found\n ->exiting"%path)
                exit()

    try:
        path_towns= "resources/towns_%s.json"%lg
        dic["towns"] = get_towns(path_towns)
    except:
        if o.debug:
            print ("  Non mandatory resource '%s' not found"%path_towns)
        dic["towns"]={}

    return dic

def open_utf8(path):
    with codecs.open(path, "r", "utf-8") as f:
        string = f.read()
    return string

def write_utf8(path, content):
    with codecs.open(path, "w", "utf-8") as w:
        w.write(content)

def translate_justext():#TODO: with big corpus, getting it only once
    dic = eval(open_utf8("resources/language_codes.json"))
    return dic

def get_lg_JT(lg_iso):
    dic_lg = translate_justext()
    lg = "unknown"

    if lg_iso in dic_lg:
        lg = dic_lg[lg_iso]

    return lg

def get_clean_html(o, lg_JT):
    if o.is_clean:
        return open_utf8(o.document_path)

    try:
        import justext
        text = open_utf8(o.document_path)
        paragraphs = justext.justext(text, justext.get_stoplist(lg_JT))
        out = ""

        for paragraph in paragraphs:
            if not paragraph.is_boilerplate:
                out+="<p>%s</p>\n"%paragraph.text

        if o.verbose:
            print ("-> Document cleaned")
    except Exception as e:
        if o.verbose:
            print (e)
            print ("** Probably Justext is missing, do 'pip install justext'")

        out = open_utf8(o.document_path)

    return out

def exist_disease(disease_info):
    return len(disease_info)

def result_filtering(results, ratio):
    for attr in ["dis_infos", "loc_infos"]:
        # TODO is there a better way?
        results[attr] = [x for x in results[attr] if x[0] >= ratio]

    if not exist_disease(results["dis_infos"]):
        return {"events": [["N", "N", "N"]]}

    return results

def process(o, resource = False, filtered=True, process_res = True, string = False):
    try:
        lg_iso = o.language
    except:
        lg_iso="unknown"

    if not string:
        string = get_clean_html(o, get_lg_JT(lg_iso))

    if not resource:
        resource = get_resource(lg_iso, o)

    results = analyze(string, resource, o)

    if filtered:
        return result_filtering(results, o.ratio)

    if process_res:
        process_results(results, options)

    return results

def valid_result(result_size, largest_ratio, threshold_ratio):
    return result_size > 0 and largest_ratio >= threshold_ratio

def print_final_result(options, results, descriptions):
    print (options.document_path)
    for info in ["dis_infos", "loc_infos"]:
        if len(results[info]) > 0:
            print (descriptions[info])
            for elem in results[info]:
                print (elem)
            print ("")

def get_final_result(results, ratio):
    res_final = {}
    for info in ["dis_infos", "loc_infos"]:
        res_final[info] = []
        for elems in results[info]:
            if elems[0] < ratio:
                break
            res_final[info].append(elems)

    return res_final

def process_results(results, options):
    ratio = float(options.ratio)
    descriptions = eval(open_utf8("resources/descriptions.json"))

    if options.debug:
        print ("-"*10, "RESULTS", "-"*10)
        print(descriptions["events"])

        for event in results["events"]:
            print("  "+" ".join(event))
        print (" ")

    if "dis_infos" not in results:
        return

    res_filtered = {}

    if valid_result(len(results["dis_infos"]), results["dis_infos"][0][0], options.ratio):
        res_filtered = get_final_result(results, options.ratio)

    if options.verbose or options.showrelevant:
        print_final_result(options, res_filtered, descriptions)

    write_utf8(options.name_out, json.dumps(res_filtered))


if __name__=="__main__":
    options = get_args()
    try: os.makedirs("tmp")
    except: pass
    start = time.clock() 
    results = process(options, resource = False, filtered = False, process_res=True)
    end = time.clock()
    print ("Time Elapse: ", round(end - start, 4))
