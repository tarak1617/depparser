from input import parse
from word2vec1 import word2vec, dictionaries
from collections import namedtuple,OrderedDict
import numpy as np
import json
import gensim
import copy
import logging


def training(fn, wordvecpath):
    if not wordvecpath:
        word2vec(fn)
        wordvecpath = './Word2Vec/vecs.bin'
    ndeprel = dictionaries(fn)
    X_lengths = np.array([])
    Arcs = namedtuple('Arcs', ['headid', 'headform', 'tailid', 'tailform', 'deprel'])
    Transition = namedtuple('Transition', ['transition', 'label'])
    with open('./dictionaries/deprel.json', 'r') as fp:
        dictionary2 = json.load(fp)
    f = open(fn, 'r')
    data = f.read()
    mode = gensim.models.Word2Vec.load(wordvecpath)
    model = mode.wv
    vecdims = mode.layer1_size
    vecdims = vecdims+11+2+2
    del mode
    Y2 = np.zeros([1, 4+ndeprel])
    X2 = np.zeros([1, vecdims*5+4])
    sid=0
    buffer1 = []
    stack = []
    arcs = []
    listofTransitions = []
    for sent in parse(data):
        del buffer1[:]
        del stack[:]
        del arcs[:]
        buffer1 = copy.deepcopy(sent)
        buffer1.append(OrderedDict(
            [("id", 0), ("form", 'root'), ("lemma", 'root'), ("upostag", 'root'), ("xpostag", 'root'), ("feats", 'root'), ("head", -1),
             ("deprel", 'root'), ("deps", 'root'), ("misc", 'root'), ]))
        flag=True
        for word in sent:
            if not pcheck(word['id'],word['head'],sent):
                del buffer1[:]
                flag=False
                break
        i=0
        while buffer1:
            transi, label = oracle(stack, buffer1, arcs)
            trans = Transition(transi, label)
            i+=1
            X,t = nn(stack, buffer1, trans, dictionary2, model, sent, arcs, vecdims, ndeprel)
            X2 = np.vstack((X2,X))
            Y2 = np.vstack((Y2,t))
            if trans.transition == 0:  # SHIFT
                stack.insert(0, buffer1[0])
                del buffer1[0]
                listofTransitions.append(trans.transition)
            elif trans.transition == 1:  # REDUCE
                del stack[0]
                listofTransitions.append(trans.transition)
            elif trans.transition == 2:  # LERFT ARC
                arcs.append(Arcs(buffer1[0]['id'], buffer1[0]['form'], stack[0]['id'], stack[0]['form'], trans.label))
                del stack[0]
                listofTransitions.append(trans.transition)
            elif trans.transition == 3:  # RIGHT ARC
                arcs.append(Arcs(stack[0]['id'], stack[0]['form'], buffer1[0]['id'], buffer1[0]['form'], trans.label))
                stack.insert(0, buffer1[0])
                del buffer1[0]
                listofTransitions.append(trans.transition)
        if flag : X_lengths = np.append(X_lengths, i)
        sid+=1
        logging.info ('vectorising sentence : '+str(sid))
    X2 = np.delete(X2, 0, axis=0)
    Y2 = np.delete(Y2, 0, axis=0)
    return X2,Y2,X_lengths


def oracle(stack, buffer1, arcs):
    global i
    if not stack:
        return 0, ""
    if not buffer1[0] :
        del buffer1[:]
        i-=1
        return 1, ""
    s0id = stack[0]['id']
    s0head = stack[0]['head']
    b0id = buffer1[0]['id']
    b0head = buffer1[0]['head']
    if b0id == s0head:
        return 2, stack[0]['deprel']
    elif s0id == b0head:
        return 3, buffer1[0]['deprel']
    elif head(stack[0], arcs) != -1 and b0head<s0head :
        return 1, ""
    return 0, ""

def head(stackc, arcs):
    for a in arcs:
        if a.headid == stackc['head']:
            return a.headid
    return -1


def nn(stack, buffer1, trans, dictionary2, model, sent, arcs, vecdims, ndeprel):
    mones = [-1] * vecdims
    ones = [1] * (vecdims-4)
    zeros = [0] * (vecdims-15)
    dep = [-1]*4
    sentenc = np.array([])
    words=["_","_","_","_","_"]
    if stack:
        words.pop(0)
        words.insert(0,stack[0])
        dep[0] = iofdeprel(rightchild(stack[0], arcs))
        dep[1] = iofdeprel(leftchild(stack[0], arcs))
        if len(stack) > 1:
            words.pop(1)
            words.insert(1,stack[1])
    if buffer1:
        words.pop(2)
        words.insert(2,buffer1[0])
        dep[2] = iofdeprel(rightchild(buffer1[0], arcs))
        dep[3] = iofdeprel(leftchild(buffer1[0], arcs))
        if len(buffer1) > 1:
            words.pop(3)
            words.insert(3,buffer1[1])
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
    sentenc = np.hstack((sentenc,dep))
    t = trans.transition
    if t > 1:
        t = np.hstack((np.eye(4)[t], np.eye(ndeprel)[iofdeprel(trans.label)-1]))
    else:
        t = np.hstack((np.eye(4)[t], np.zeros(ndeprel)))

    return sentenc, t

def D(key, dic):
    if dic.get(key): return dic[key]
    return -1;


def featureids(feats1, dic):
    f=[-1]*11
    if feats1['cat'] in dic: f[0] = dic[feats1['cat']]
    if feats1['gen'] in dic: f[1] = dic[feats1['gen']]
    if feats1['num'] in dic: f[2] = dic[feats1['num']]
    if feats1['pers'] in dic: f[3] = dic[feats1['pers']]
    if feats1['case'] in dic: f[4] = dic[feats1['case']]
    if feats1['vib'] in dic: f[5] = dic[feats1['vib']]
    if feats1['tam'] in dic: f[6] = dic[feats1['tam']]
    if feats1['chunkId'] in dic: f[7] = dic[feats1['chunkId']]
    if feats1['chunkType'] in dic: f[8] = dic[feats1['chunkType']]
    if feats1['stype'] in dic: f[9] = dic[feats1['stype']]
    if feats1['voicetype'] in dic: f[0] = dic[feats1['voicetype']]
    return f

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

def iofdeprel(ele):
    with open('./dictionaries/deprel.json', 'r') as fp:
        dict = json.load(fp)
    if ele in dict: return dict[ele]
    return -1

def pcheck(id1,id2,sentence):
    flag=True
    if id2>id1:
        for words in sentence[id1:id2-1]:
            if words['head'] > id2 or words['head'] < id1:
                flag=False
                break
    if id1>id2:
        for words in sentence[id2:id1-1]:
            if words['head'] > id1 or words['head'] < id2 :
                flag=False
                break
    return flag

# fn = 'trainfile.conll'
# numpypath = './numpysave/'
# X2new, Y2new, X_lengthsnew = training(fn)
# print (X2new.shape)
# np.save(numpypath+'X2new', X2new)
# print (X_lengthsnew.shape)
# X_lengthsnew = X_lengthsnew.astype(int)
# np.save(numpypath+'X_lengthsnew', X_lengthsnew)
# print (Y2new.shape)
# Y2new = Y2new.astype(int)
# np.save(numpypath+'Y2new', Y2new)
