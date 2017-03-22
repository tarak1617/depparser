import gensim
import logging
import json
from input import parse
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)
def word2vec(fn):
    f = open(fn, 'r')
    wr = open('./tmpdata/w2vecinput.txt', 'w')
    data = f.read()
    for sent in parse(data):
        for buffer in sent:
            wr.write(buffer['form'])
            wr.write(' ')
        wr.write('\n')
    wr.close()
    fname = open('./tmpdata/vecs.bin','w')
    sentences = gensim.models.word2vec.LineSentence('./tmpdata/w2vecinput.txt')
    model = gensim.models.Word2Vec(sentences, min_count=1,window=5, size=128, sg=1, iter=500)

    model.init_sims(replace=True)
    model.save(fname)

def dictionaries(fn):
    f = open(fn, 'r')
    i=0
    dictionary = dict()
    data = f.read()
    for sent in parse(data):
        for bufer in sent:
            if bufer['deprel'] not in dictionary :
                i=i+1
                dictionary[bufer['deprel']]=i
    ndeprels = len(dictionary)
    with open('./tmpdata/deprel.json', 'w') as fp:
        json.dump(dictionary, fp)
    for sent in parse(data):
        for bufer in sent:
            if bufer['xpostag'] not in dictionary :
                i=i+1
                dictionary[bufer['xpostag']]=i
            if bufer['upostag'] not in dictionary :
                i=i+1
                dictionary[bufer['xpostag']]=i
            for strin in bufer['feats'].keys():
                if bufer['feats'][strin] not in dictionary:
                    dictionary[bufer['feats'][strin]] = i
                    i += 1
    with open('./tmpdata/all.json', 'w') as fp:
        json.dump(dictionary, fp)
    return ndeprels

