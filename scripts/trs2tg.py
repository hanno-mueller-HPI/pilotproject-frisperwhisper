#Version 20.11.2024 FS, zuerst VH 2019

import glob
import re
import sys
import os
import copy
#import pickle
import itertools
import collections.abc

if len(sys.argv) < 2:
    print("Usage: python trs2tg.py <directory>")
    sys.exit(1)

input_dir = sys.argv[1]
if not os.path.isdir(input_dir):
    print(f"Error: {input_dir} is not a valid directory.")
    sys.exit(1)

def getmeta(metar, regex,span1,span2):
    rs = re.compile(regex, re.DOTALL).search(metar)
    if rs:
        out = rs.group()[span1:span2]
    else:
        out = "na"
    return out

def check(metar, regex,a,b):
    check = re.compile(regex, re.DOTALL).findall(metar)
    check = [element[a:b] for element in check]
    return check

def replace(regex, bywhat, inwhere):
    regex = re.compile(regex, re.DOTALL)
    newstring = regex.sub(bywhat, inwhere)
    return newstring

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

####Start####
listspk = []


trs_files = glob.glob(os.path.join(input_dir, "*.trs"))
for file in trs_files:
    with open (file, "rt", encoding="ISO-8859-1") as trans:
        print(file)
        trans = trans.read()
        #Beseitigen überlappender Rede
        odict = {}
        overlapping = check(trans, r"<Turn speaker=\"spk\d* spk\d*\".*?</Turn>", None, None)
        #print(overlapping)
        for oturn in overlapping:
            a = getmeta(oturn, "<Turn.*?>", None, None)
            odict[a] = oturn
        for key in odict.keys():
            oendtime = getmeta(odict[key], r"endTime=\".*?\"", 7, None)            
            odict[key] = replace(r"</Turn>", "<Sync time" + oendtime + "/>\n</Turn>", odict[key])
            #print(oendtime)
            odict[key] = replace(r"<Sync time(=\"\d*\.?\d*\")(/>\n<Who nb=\"1\")(/>.*?\n<Who nb=\"2\")/>", "<Sync time\\1\\2 startTime\\1\\3 startTime\\1/>", odict[key])
            #print(odict)
            odict[key] = replace(r"(<Who nb=\"1\" startTime=\".*?\")/(>.*?<Who nb=\"2\" startTime=\".*?\")/(>.*?\n<Sync time)(=\"\d*\.?\d*\")", "\\1 endTime\\4\\2 endTime\\4\\3\\4", odict[key])
            #print(odict)
            who1 = getmeta(key, "<Turn speaker=\"spk\d*", 15, None)
            who2 = getmeta(key, " spk\d*\" start", 1, -7)
            odict[key] = replace(r"Who nb=\"1\"", "Turn speaker=\"" + who1 + "\"", odict[key])
            odict[key] = replace(r"Who nb=\"2\"", "Turn speaker=\"" + who2 + "\"", odict[key])
            #print(odict)
            odict[key] = replace(r"(<Sync.*?>)\n(<Turn.*?>)", "\\2\n\\1", odict[key])
            odict[key] = replace(r"<Turn speaker=\"spk\d* spk\d*\".*?>\n", "", odict[key])
            odict[key] = replace(r"\n<Turn", "\n</Turn>\n<Turn", odict[key])
            #Löschen von Sync
            odict[key] = replace(r"\n<Sync time.*?>", "", odict[key])
            #Ersetzen in trs
            trans = replace(key + r".*?</Turn>", odict[key], trans)
        #print(odict)
    #print(trans)
        #Kopfzeile in grid
        lastend = check(trans, r"endTime=\".*?\">", 9, -2)[-1]
        #print(lastend)
        text = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n\nxmin = 0 \nxmax = " + lastend + " \n" + "tiers? <exists> \n"
        listspk = check(trans, r"<Speaker id=\"spk\d\"", 12, None)
        text = text + "size = " + str(len(listspk)) + " \nitem []: \n"
        #print (listspk)
        #Auflistung der Turns
        #print(trans, "\n\n\n")
            
                          
        for spk in listspk:
            turnsperspk = check(trans, r"<Turn[^\n]*?speaker=" + spk + r".*?</Turn>", None, None)
            #print(turnsperspk) #ok
            #seg>turns
            for tnum in range(len(turnsperspk)-1):
                #print(tnum)
                #print(turnsperspk[tnum])
                tend = getmeta(turnsperspk[tnum], "endTime=\".*?\"", 9, -1)
                #print(tend)
                tsta = getmeta(turnsperspk[tnum+1], "startTime=\".*?\"", 11, -1)
                #print(tend, "\t", tsta)
                if tend != tsta:
                    provturn = copy.deepcopy([turnsperspk[tnum], "<Turn speaker=" + spk + " startTime=\"" + tend + "\" endTime=\"" + tsta + "\">\n\n</Turn>"])
                    #print(provturn)
                    turnsperspk[tnum] = copy.deepcopy(provturn)
                    #print(len(turnsperspk))
            if len(turnsperspk) > 0:
                laendspk = getmeta(turnsperspk[-1], "endTime=\".*?\"", 9, -1)
                if laendspk != lastend:
                    turnsperspk.append("<Turn speaker=" + spk + " startTime=\"" + laendspk + "\" endTime=\"" + lastend + "\">\n\n</Turn>")
                fistaspk = getmeta(str(turnsperspk[0]), "startTime=\".*?\"", 11, -1)
                if fistaspk != "0":
                    turnsperspk.insert(0, "<Turn speaker=" + spk + " startTime=\"" + "0" + "\" endTime=\"" + fistaspk + "\">\n\n</Turn>")
                turnsperspk = list(flatten(turnsperspk))
            #print(turnsperspk)
            
            for tu in turnsperspk:
                tendtime = getmeta(tu, r"endTime=\".*?\"", 7, None)
                tu2 = replace(r"</Turn>", "<Sync time" + tendtime + "/>\n</Turn>", tu)
                syncs = check(tu2, "<Sync time.*?/>", None, None)
                #print(syncs) #nk
                if len(syncs)>1:
                    #print(syncs) #nk
                    for onesync in range(len(syncs)):
                        syncs[onesync] = replace(r"<Sync time=\"(.*?)\"/>", "\\1", syncs[onesync])
                        #print(syncs[onesync])
                    #print(syncs) #nk
                    for onesync in range(len(syncs)-1):
                        #print(syncs[onesync])
                        newturn = "<Turn speaker=" + spk + " startTime=\"" + syncs[onesync] + "\" endTime=\"" + syncs[onesync+1] + "\">"
                        #print(newturn) #nk
                        tu2 = replace(r"<Sync time=\"" + syncs[onesync] + "\"/>", newturn + "\n<Sync time=\"" + syncs[onesync] + "\"/>", tu2)
                        tu2 = replace(r"<Turn[^\n]*?\n<Turn", "<Turn", tu2)
                        tu2 = replace(r"(<Turn[^\n]*?\n)<Sync time=\"\d*\.?\d*?\"/>\n", "\\1", tu2)
                        #print(check(tu, "<Sync time=\"\d*\.?\d*?\"/>", None, None))
                    #print(tu2)
                for onesync in range(len(syncs)):
                    segturns = []
                    tu2 = replace(r"<Sync time=\"\d*\.?\d*?\"/>", "</Turn>", tu2)
                    tu2 = replace(r"<Turn", "</Turn>\n<Turn", tu2)
                    tu2 = replace(r"</Turn>\n</Turn>", "</Turn>", tu2)
                    #print(tu2)
                segturns = copy.deepcopy(check(tu2, r"<Turn.*?</Turn>", None, None))
                #print(segturns)
                turnsperspk[(turnsperspk.index(tu))] = segturns
                #print(turnsperspk)
            turnsperspk = list(itertools.chain.from_iterable(turnsperspk))
            #print(turnsperspk, "\n")
            if len(turnsperspk) != 0:
                text = text + "    item [" + str(listspk.index(spk) + 1) + "]:\n        class = \"IntervalTier\" \n"
                text = text + "        name = " + spk + " \n"
                text = text + "        xmin = " + getmeta(turnsperspk[0], r"startTime=\".*?\"", 11, -1) + " \n"
                text = text + "        xmax = " + getmeta(turnsperspk[-1], r"endTime=\".*?\"", 9, -1) + " \n"
                text = text + "        intervals: size = " + str(len(turnsperspk)) + " \n"
        #print(text)
                x = 1
                #print(turnsperspk)#ok
                for turn in turnsperspk:
                    #print(turn) #ok
                    text = text + "        intervals [" + str(x) + "]:\n"
                    start = getmeta(turn, r"startTime=\".*?\"", 11, -1)
                    text = text + "            xmin = " + start + " \n"
                    end = getmeta(turn, r"endTime=\".*?\"", 9, -1)
                    if float(end)< float(start):
                        end=start
                    text = text + "            xmax = " + end + " \n"
                    #Hier löscht er alles, was in spitzen Klammern steht. Bei Bedarf auskommentieren. Oder was wird daraus?
                    turn = replace (r"<Sync.*?>", "", turn)
                    turn = replace (r"<Event desc=\"(.*?)\" .*?>", "(\\1)", turn)
                    turn = replace (r"\n", " ", turn)
                    turn = replace (r" +", " ", turn)
                    turntxt = getmeta(turn, "\">.*</Turn>", 3,-8)
                    text = text + "            text = \"" + turntxt + "\" \n"
                    x = x+1
            else:
                text = text + "    item [" + str(listspk.index(spk) + 1) + "]:\n        class = \"IntervalTier\" \n"
                text = text + "        name = " + spk + " \n"
                text = text + "        xmin = 0 \n"
                text = text + "        xmax = " + lastend + " \n"
                text = text + "        intervals: size = 1 \n"
        #print(text)
                x = 1
                #print(turnsperspk)#ok
        
                #print(turn) #ok
                text = text + "        intervals [" + str(x) + "]:\n"
                start = "0"
                text = text + "            xmin = " + start + " \n"
                end = lastend
                text = text + "            xmax = " + end + " \n"
                text = text + "            text = \"\" \n"
                x = x+1
            #print(text)
        nfile = replace (r"(.*?)(\.trs)", r"\1" + ".TextGrid", file)
        with open (nfile, "wt", encoding="ISO-8859-1") as newname:
                 newname.write(text)
     

