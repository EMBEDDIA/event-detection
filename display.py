#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import re

f = open(sys.argv[1])
lignes = f.readlines()
f.close()

os.system("rm sandbox/tmp_relevant/*")
cpt = 0
for l in lignes:
    if "/home" in l:
        cpt += 1
        l = re.sub("\n", "", l)
#    cmd = "google-chrome %s"%l
#    os.system(cmd)
        nom_fic = re.split("/", l)[-1]
        nom_out = "sandbox/tmp_relevant/%s" % nom_fic
        cmd2 = "cp %s %s" % (l, nom_out)
        os.system(cmd2)
        if cpt == 15:
            break
    elif l[0] == "[":
        f = open(nom_out)
        data = f.read()
        f.close()
        L = eval(l)
        desc = "<p>Disease : "+L[1]+" Substring (%s) : " % str(L[0])+L[2]
        data_out = desc + "\n" + data
        w = open(nom_out, "w")
        w.write(data_out)
        w.close()
