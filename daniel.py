#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import json
from tools import *
from rstr_max import *
from tools_karkkainen_sanders import *
import codecs
import sys
import os
import re
import glob
import math
sys.path.append('./rstr_max')

# Maybe a bottle neck
# normalizes the distance between the substring and the closest end of the text


def get_normalized_pos(ss, text):
    l_dist = [m.start() for m in re.finditer(re.escape(ss), text)]
#    l_dist = [min(x, len(text)-x) for x in l_dist]
    return [float(x)/len(text) for x in l_dist]

# check for bottle-neck
# TODO: looks kinda complicated, will check again


def exploit_rstr(r, rstr, s_id_txt):
    desc = []
    # if paragraph structure is weak
    weak_struct = len(s_id_txt) == 1
    for (offset_end, nb), (l, start_plage) in r.items():
        ss = rstr.global_suffix[offset_end-l:offset_end]
        s_occur = set()

        for o in range(start_plage, start_plage+nb):
            s_occur.add(rstr.idxString[rstr.res[o]])

        # how many repetitions in different paragraphs
        inter = s_occur.intersection(s_id_txt)

        #  repeated in text and present in disease list
        has_inter = (len(inter) > 1 and len(s_occur) > len(inter))

        # needs to be in 1st paragraph
        weak_and_repeat = (weak_struct and 0 in s_occur)
        # condition 1 : repetion pattern found
        # condition 2 : weak structure, relax the constraint
        if has_inter or weak_and_repeat:
            NE_ids = [x - len(s_id_txt) for x in s_occur.difference(s_id_txt)]

            if len(inter) > 1:
                # gives too much importance to the footer (noise)
                #l_dist = [min(pos, len(s_id_txt) - pos - 1) for pos in inter]
                l_dist = inter
            else:  # short documents
                l_dist = get_normalized_pos(ss, rstr.global_suffix)

            l_dist = [round(x, 5) for x in l_dist]
            desc.append([ss, NE_ids, sorted(l_dist)])
    # returns triplest :  substring + Named Entity ID + positions
    return desc


def get_score(ratio, dist, s_id_txt):
    if ratio == 1:
        ratio = 0.99
    # not the same score for repetitions in header, footer elsewhere
    if dist[1] < 2:  # repetition in the header
        return min(ratio + 0.1, 1)  # bonus
    elif dist[0] == 0:  # title + somewhere in the document
        return min(ratio + 0.05, 1)  # bonus
    elif dist[0] <= 2:  # header +rest
        return ratio
        # return pow(ratio, 1+dist[0]*math.log(dist[1]))
#    elif len(s_id_txt)-1 in dist or len(s_id_txt)-2 in dist:#footer repetition
#        return pow(ratio, 1+dist[0]*math.log(dist[1]))
    else:  # malus
        return pow(ratio, 1+dist[0]*dist[1])

# TODO: for later uses
# def substring_score(entity_name, ss, distances, loc):
#     ratio = float(len(ss))/len(entity_name)
#
#     if ss[0].lower() != entity_name[0].lower():
#         if loc:
#             # for country names the first character should not change
#             ration = max(0, ratio - 0.2)    #penalty
#         else:
#             if len(entity_name) < 6 and ratio <1:
#                 ratio = max(0, ratio - 0.1) #penalty
#
#     score = get_score(ratio, distances)
#
#     return [score, entity_name, ss, distances]

# check here for bottle-neck
# TODO: try to remove nested loop later


def filter_desc(desc, l_rsc, s_id_txt, loc=False):
    out = []
    for ss, dis_list, distances in desc:
        for id_dis in dis_list:
            entity_name = l_rsc[id_dis]
            ratio = float(len(ss))/len(entity_name)

            if ss[0].lower() != entity_name[0].lower():
                if loc:
                    # for country names the first character should not change
                    ratio = max(0, ratio-0.2)  # penalty
                else:
                    ratio = max(0, ratio-0.1)  # penalty

            score = get_score(ratio, distances, s_id_txt)
            out.append([score, entity_name, ss, distances])
        # entity_name = list(map(lambda x: l_rsc[int(x)], dis_list))
        # test = list(map(lambda x: substring_score(x, ss, distances, loc), entity_name))

        # out.append(test)
    return sorted(out, reverse=True)


def get_desc(string, rsc, loc=False):
    s_id_txt = set()
    rstr = Rstr_max()
    cpt = 0
    l_rsc = list(rsc.keys())

    for s in string:
        rstr.add_str(s)
        s_id_txt.add(cpt)
        cpt += 1

    for r in l_rsc:
        rstr.add_str(r)

    r = rstr.go()  # ???? should name different perhap
    desc = exploit_rstr(r, rstr, s_id_txt)

    return filter_desc(desc, l_rsc, s_id_txt, loc)


def zoning(string, options):
    z = re.split("<p>", string)

    if len(z) == 1:
        z = re.split("\n", string)

    z = [x for x in z if x != ""]

    if len(z) < 3:  # insufficient paragraph/linebreaks structure
        sentences = re.split("\. |\</p>", string)
        sentences = [x for x in sentences if len(x) > 2]

        if len(sentences) < 5:  # very short article
            z = [string]
        elif len(z) == 2:  # Title may have been extracted
            part = int(len(z[1])/2)
            z = [z[0], z[1][:part], z[1][part:]]
        else:  # No usable structure
            part = int(len(string)/3)
            z = [string[:part], string[part:part*2], string[part*2:]]
    elif options.language != "zh":
        if len(re.findall("\.[ <]", z[1])) < 2:  # reorganize header
            z = [z[0], z[1]+z[2]] + z[2:]

    if options.debug:
        for zone in z:
            print(re.sub("\n", "***", zone[:70]))
            print("")
        d = input("Zoning ended, proceed to next step ?")

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
        print("  ", round(elem[0], 2), " \t", elem[1], "\t", elem[2])

# TODO find a better way to write this


def get_town(town_infos, loc, ratio):
    if town_infos:
        for town in town_infos:
            if town[0] < ratio:
                break
            loc.append((town[1], town[0]))


def get_event(dis_infos, loc):
    return [(dis[1], loc) for dis in dis_infos[:1] if dis_infos]


def analyze(string, resource, options):
    zones = zoning(string, options)
    dis_infos = get_desc(zones, resource["diseases"])

    if len(dis_infos) <= 0:
        # print "No infos on diseases"
        return {"events": [], "dis_infos": dis_infos, "loc_infos": []}

    loc_infos = get_desc(zones, resource["locations"], True)

    if options.verbose:
        print("Top 5 name entities for diseases: ")
        print_top5(dis_infos)

        print("Top 5 name entities for locations: ")
        print_top5(loc_infos)

    if len(loc_infos) == 0 or loc_infos[0][0] < 0.5:
        loc = get_implicit_location(resource, options)
    else:
        loc = loc_infos[0][1]

    town_infos = get_desc(zones, resource["towns"], True)

    get_town(town_infos, loc, options.ratio)

    events = get_event(dis_infos, loc)

    return {"events": events, "dis_infos": dis_infos, "loc_infos": loc_infos}


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
        path = "resources/%s_%s.json" % (rsc_type, lg)
        if os.path.exists(path):
            try:
                dic[rsc_type] = eval(open_utf8(path))
            except Exception as e:
                if rsc_type in mandatory_rsc:
                    print("\n  Problem with resource %s :" % path)
                    print(e)
                    exit()
                else:
                    dic[rsc_type] = {}
        else:
            if rsc_type in mandatory_rsc:
                print("  Ressource '%s' not found\n ->exiting" % path)
                exit()

    try:
        path_towns = "resources/towns_%s.json" % lg
        dic["towns"] = get_towns(path_towns)
    except:
        if o.debug:
            print("  Non mandatory resource '%s' not found" % path_towns)
        dic["towns"] = {}
# improved_dic = {}#does not work with russian
# for key, val in dic["diseases"].items():
# if key.upper()==key:
##        improved_dic[key] =val
# else:
##        improved_dic[key.lower()] =val
##        improved_dic[key.capitalize()] =val
# dic["diseases"]=improved_dic
    return dic


def open_utf8(path):
    with codecs.open(path, "r", "utf-8") as f:
        string = f.read()
    return string


def write_utf8(path, content):
    with codecs.open(path, "w", "utf-8") as w:
        w.write(content)


def translate_justext():  # TODO: with big corpus, getting it only once
    dic = eval(open_utf8("resources/language_codes.json"))
    return dic


def get_lg_JT(lg_iso):
    dic_lg = translate_justext()
    lg = "unknown"

    if lg_iso in dic_lg:
        lg = dic_lg[lg_iso]

    return lg


def get_clean_html(o, lg_JT):
    if o.isnot_clean == False:
        return open_utf8(o.document_path)

    try:
        import justext
        text = open_utf8(o.document_path)
        pars = justext.justext(text, justext.get_stoplist(lg_JT))
        out = "\n".join(
            ["<p>%s</p>" % p.text for p in pars if not p.is_boilerplate])

        if o.verbose:
            print("-> Document cleaned")
    except Exception as e:
        if o.verbose:
            print("\n** %s" % str(e))
            print("** may be 'pip install justext' will be needed")
            print("** otherwise remove the -i option\n")
        exit()

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


def process(o, resource=False, filtered=True, process_res=True, string=False):
    try:
        lg_iso = o.language
    except:
        lg_iso = "unknown"

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
    print(options.document_path)
    if results == {}:
        return
    for info in ["dis_infos", "loc_infos"]:
        if len(results[info]) > 0:
            print(descriptions[info])
            for elem in results[info]:
                print(elem)
            print("")


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
        print("-"*10, "RESULTS", "-"*10)
        print(descriptions["events"])

        for event in results["events"]:
            print("  "+" ".join(event))
        print(" ")

    if "dis_infos" not in results:
        return

    res_filtered = {}

    if valid_result(len(results["dis_infos"]), results["dis_infos"][0][0], options.ratio):
        res_filtered = get_final_result(results, options.ratio)

    if options.verbose or options.showrelevant:
        print_final_result(options, res_filtered, descriptions)

    write_utf8(options.name_out, json.dumps(res_filtered))


if __name__ == "__main__":
    options = get_args()
    try:
        os.makedirs("tmp")
    except:
        pass
    start = time.clock()
    results = process(options, resource=False,
                      filtered=False, process_res=True)
    end = time.clock()
    print("Time Elapse: ", round(end - start, 4))
