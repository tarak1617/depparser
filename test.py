from collections import namedtuple, OrderedDict
from input import parse
import tensorflow as tf
import numpy as np
import json
import copy
import gensim
import subprocess
import re
import logging

def parsing(file, wordvecpath, numpypath, ckptpath,Yp, Yd, H, X, XL, Hin, keep_prob):
    logging.info('parsing started')
    f = open(file, 'r')
    with open('./dictionaries/deprel.json', 'r') as fp:
        deps = json.load(fp)
    ndeprel = len(deps)
    with open('./dictionaries/all.json', 'r') as fp:
        dictionary2 = json.load(fp)
    mode = gensim.models.Word2Vec.load(wordvecpath)
    vecdims = mode.layer1_size
    vecdims = vecdims + 11 + 2 + 2
    model = mode.wv
    Arcs = namedtuple('Arcs', ['headid', 'headform', 'tailid', 'tailform', 'deprel'])
    Transition = namedtuple('Transition', ['transition', 'label'])
    writefile = open('output.conll', 'w')
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess,ckptpath)
    inH = np.load(numpypath)
    sid=0
    buffer1 = []
    stack = []
    arcs = []
    data = f.read()
    for sent in parse(data):
        del buffer1[:]
        del stack[:]
        del arcs[:]
        buffer1 = copy.deepcopy(sent)
        buffer1.append(OrderedDict(
            [("id", 0), ("form", 'root'), ("lemma", 'root'), ("upostag", 'root'), ("xpostag", 'root'), ("feats", 'root'), ("head", -1),
             ("deprel", 'root'), ("deps", 'root'), ("misc", 'root'), ]))
        while buffer1:
            transi, label, inH = oracle(stack, buffer1, arcs, dictionary2, model,sent,sess,inH, vecdims,X,XL,Hin,keep_prob, H, Yp,Yd)
            # print(label)
            trans = Transition(transi, label)
            if trans.transition == 0:  # SHIFT
                stack.insert(0, buffer1[0])
                del buffer1[0]
            elif trans.transition == 1:  # REDUCE
                if stack : del stack[0]
            elif trans.transition == 2:  # LERFT ARC
                if stack :
                    arcs.append(Arcs(buffer1[0]['id'], buffer1[0]['form'], stack[0]['id'], stack[0]['form'], trans.label))
                    del stack[0]
                else:
                    stack.insert(0, buffer1[0])
                    del buffer1[0]
            elif trans.transition == 3:  # RIGHT ARC
                if stack and buffer1:
                    arcs.append(Arcs(stack[0]['id'], stack[0]['form'], buffer1[0]['id'], buffer1[0]['form'], trans.label))
                    stack.insert(0, buffer1[0])
                    del buffer1[0]
                else :
                    stack.insert(0, buffer1[0])
                    del buffer1[0]
            else :
                stack.insert(0, buffer1[0])
                del buffer1[0]
        # print(arcs)
        # print(sent)
        attacharclabel(sent, arcs)
        # print(sent)
        for s in sent:
            reverseparse(s['id'], s['form'], s['lemma'], s['upostag'], s['xpostag'], s['feats'], s['head'], s['deprel'], s['deps'], s['misc'], writefile)
        writefile.write("\n")
        sid+=1
        logging.info('parsing sentence : '+str(sid))
    sess.close()
    writefile.close()
    # removelastline("output.conll")
    logging.info('parsing complete')
    las,uas,la = evaluate()
    return las,uas,la

def oracle(stack, buffer1, arcs, dictionary2, model,sent,sess,inH, vecdims,X,XL,Hin,keep_prob, H, Yp,Yd):
    mones = [-1] * vecdims
    ones = [1] * (vecdims - 4)
    zeros = [0] * (vecdims - 15)
    dep = [-1] * 4
    sentenc = np.array([])
    words = ["_", "_", "_", "_", "_"]
    if stack:
        words.pop(0)
        words.insert(0, stack[0])
        dep[0] = iofdeprel(rightchild(stack[0], arcs))
        dep[1] = iofdeprel(leftchild(stack[0], arcs))
        if len(stack) > 1:
            words.pop(1)
            words.insert(1, stack[1])
    if buffer1:
        words.pop(2)
        words.insert(2, buffer1[0])
        dep[2] = iofdeprel(rightchild(buffer1[0], arcs))
        dep[3] = iofdeprel(leftchild(buffer1[0], arcs))
        if len(buffer1) > 1:
            words.pop(3)
            words.insert(3, buffer1[1])
            if len(buffer1) > 2:
                words.pop(4)
                words.insert(4, buffer1[2])
    for w in words:
        if w == '_':
            sentenc = np.hstack((sentenc, mones))
        elif w['form'] == 'root':
            sentenc = np.hstack((sentenc, ones, D(w['upostag'], dictionary2), D(w['xpostag'], dictionary2), w['id'], len(sent)))
        elif w['form'] in model.vocab:
            sentenc = np.hstack((sentenc, model[w['form']], featureids(w['feats'], dictionary2),D(w['upostag'], dictionary2), D(w['xpostag'], dictionary2), w['id'], len(sent)))
        elif w['form'] is not None:
            sentenc = np.hstack((sentenc, zeros, featureids(w['feats'], dictionary2), D(w['upostag'], dictionary2), D(w['xpostag'], dictionary2), w['id'], len(sent)))
        else:
            sentenc = np.hstack((sentenc, mones))
    sentenc = np.hstack((sentenc, dep))
    line = sentenc.reshape([1, -1, vecdims*5+4])
    t,depre, inH = sess.run([Yp,Yd,H], feed_dict={X: line, XL: [1], Hin: inH, keep_prob:1.0})
    dl = riofdeprel(int(depre))
    # print (dl,t,depre, inH)
    return int(t), dl, inH

def D(key, dic):
    if dic.get(key): return dic[key]
    return -1;

def reverseparse(id, form, lemma, upostag, xpostag, feats, head, deprel, deps, misc, f):
    filewrite(f, str(id))
    filewrite(f, str(form))
    filewrite(f, str(lemma))
    filewrite(f, str(upostag))
    filewrite(f, str(xpostag))
    str1=""
    for feat in feats.iteritems():
        if feat[1]:
            str1 += (feat[0] + '-' + feat[1] + '|')
    str1=str1[:-1]
    filewrite(f, str(str1))
    filewrite(f, str(head))
    filewrite(f, str(deprel))
    filewrite(f, str(deps))
    f.write(str(misc))
    f.write("\n")
def filewrite(f, str):
    if str : f.write(str)
    else: f.write('_')
    f.write('\t')
def riofdeprel(id):
    with open('./dictionaries/deprel.json', 'r') as fp:
        dic = json.load(fp)
    dic = dict((v, k) for k, v in dic.items())
    if id in dic:
        return dic[id]
    else:
        return ""
def rightchild(stackc, arcs):
    id=-1
    deprel=""
    for a in arcs :
        if a.headid == stackc['id'] and a.tailid > stackc['id']:
            if id==-1 :
                id=a.tailid
                deprel=a.deprel
            else :
                if id < a.tailid :
                    id = a.tailid
                    deprel = a.deprel
    return deprel

def leftchild(stackc, arcs):
    id=-1
    deprel=""
    for a in arcs :
        if a.headid == stackc['id'] and a.tailid < stackc['id'] :
            if not id :
                id = a.tailid
                deprel = a.deprel
            else :
                if id > a.tailid :
                    id = a.tailid
                    deprel = a.deprel
    return deprel



def headd(stackc, arcs):
    for a in arcs:
        if a.headid == stackc['head']:
            return a.headid, a.deprel
    return None,""

def featureids(feats1, dic):
    f=[]
    for k in feats1 :
        if k is not None : f.append(D(feats1[k], dic))
        else: f.append(-1)
    return f

def iofdeprel(deprel):
    with open('./dictionaries/deprel.json', 'r') as fp:
        dic = json.load(fp)
    if deprel in dic:
        return dic[deprel]
    else:
        return -1

def attacharclabel(sent, arcs):
    for s in sent:
        s['head'] =0
        s['deprel']='root'
    for a in arcs:
        sent[a.tailid-1]['head']=a.headid
        sent[a.tailid-1]['deprel'] = a.deprel
    return sent

def removelastline(f):
    readfile = open(f)
    lines = readfile.readlines()
    readfile.close()
    w = open(f,'w')
    w.writelines([i for i in lines[:-1]])
    w.close()
    return

def evaluate():
    var = './'
    pipe = subprocess.Popen(["perl", var+"eval07.pl", "-q", "-g", "test100.conll", "-s", "output.conll"], stdout=subprocess.PIPE)
    out = pipe.stdout.read()
    las, uas, la = re.findall('[\d]+[.][\d]+', str(out))
    # outfile = open('logs.txt', 'a')
    # outfile.write(str(out))
    # print (out)
    return las,uas,la


# ckptpath = "./classifiersave/classifier_0.01_10_"
# numpypath = "./numpysave/outH_0.01_10_.npy"
# testfile = 'test100.conll'
# # # f = open(testfile, 'r')
# print (parsing(testfile, numpypath, ckptpath))




