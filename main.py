import numpy as np
import tensorflow as tf
from traindata import training
from test import parsing
import logging
import re
import os
import csv
import gensim
import json
import argparse

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
logging.info('logger started')

def traindata1(fn, wordvecpath):
    X2, Y2, X_lengths = training(fn, wordvecpath)
    print(X2.shape)
    np.save(numpypath+'X2', X2)
    X_lengths = X_lengths.astype(int)
    print(X_lengths.shape)
    np.save(numpypath+'X_lengths', X_lengths)
    print(Y2.shape)
    Y2 = Y2.astype(int)
    np.save(numpypath+'Y2', Y2)

def train_classifier(numpypath, ckptpath, lr, iters):
    logging.info('classifier training started')
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        inH = np.zeros([BATCHSIZE, CELLSIZE * NLAYERS])
        X_data = np.load(numpypath + 'X2.npy')
        Y_data = np.load(numpypath + 'Y2.npy')
        X_lengths = np.load(numpypath + 'X_lengths.npy')
        s = X_lengths.size
        iters=iters*s+1
        epoch =0
        while epoch<iters:
            X_, X_len, Y_b= next_batch(BATCHSIZE, X_data, X_lengths, Y_data)
            dic = {X : X_, XL : X_len,Y_ : Y_b, Hin : inH, keep_prob : 0.75}
            _,outH = sess.run([train_step, H,], feed_dict=dic)
            inH = outH
            epoch+=1
            logging.info('epoch no : '+str(epoch)+', iterations : '+str(epoch/s))
        np.save(numpypath + "outH_" + str(lr) + "_" + str(int(epoch / s)) + "_", outH)
        saver.save(sess, ckptpath + "classifier_" + str(lr) + "_" + str(int(epoch / s)) + "_")
        logging.info('checkpoint save of epoch ' + str(epoch))
        logging.info('classifier training complete')

i=0
k=0
def next_batch(bs, X_data, X_lengths, Y_data):
    global i,k
    if k == X_lengths.size :
        k=0
        i=0
    XLres = X_lengths[k:k+bs]
    k+=bs
    ma = np.amax(XLres)
    Xres = np.zeros([1, CELLSIZE])
    Yres = np.zeros([1, NCLASSES])
    for t in XLres:
        temp=X_data[i:i+t]
        temp2 = Y_data[i:i + t]
        i+=t
        if t<ma and temp.size:
            npad = ((0, ma - t), (0, 0))
            temp = np.pad(temp, pad_width=npad, mode='constant', constant_values=0)
            temp2 = np.pad(temp2, pad_width=npad, mode='constant', constant_values=0)
        Xres = np.vstack((Xres,temp))
        Yres = np.vstack((Yres, temp2))
    Xres = np.delete(Xres, 0, axis=0)
    Yres = np.delete(Yres, 0, axis=0)
    Xres = Xres.reshape([bs, ma, CELLSIZE])
    return Xres, XLres, Yres

ap = argparse.ArgumentParser()
ap.add_argument("-t","--train", help="input file for training")
ap.add_argument("-p","--parse", help="input file for parsing")
ap.add_argument("-w","--wordvecs", help="input file for wordvectors/wordembeddings")
ap.add_argument("-lr","--learningrate", help="learning rate for training classifier", default=0.001)
ap.add_argument("-its","--iterations", help="number of iterations for training classifier", default=10)
args = ap.parse_args()
traindatafile = args.train
testfile = args.parse
lr = float(args.learningrate)
iters = int(args.iterations)
numpypath = './tmpdata/'
ckptpath = './tmpdata/'


if traindatafile : traindata1(traindatafile, args.wordvecs)
if args.wordvecs : wordvecpath=args.wordvecs
else: wordvecpath = './tmpdata/vecs.bin'
mode = gensim.models.Word2Vec.load(wordvecpath)
vecdims = mode.layer1_size
vecdims = vecdims+11+2+2
with open('./tmpdata/deprel.json', 'r') as fp:
    depreldic = json.load(fp)
ndeprel=len(depreldic)
tf.variable_scope('tfl', reuse=True)
BATCHSIZE = 1
CELLSIZE = vecdims * 5 + 4
NCLASSES = ndeprel + 4
NLAYERS = 3
X = tf.placeholder(tf.float32, [None, None, CELLSIZE])
Y_ = tf.placeholder(tf.int32, [None, NCLASSES])
XL = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)
Hin = tf.placeholder(tf.float32, [None, CELLSIZE * NLAYERS])
cell = tf.contrib.rnn.GRUCell(CELLSIZE)
mcell = tf.contrib.rnn.MultiRNNCell([cell] * NLAYERS, state_is_tuple=False)
Hr, H = tf.nn.dynamic_rnn(cell=mcell, inputs=X, sequence_length=XL, initial_state=Hin, dtype=tf.float32,
                          time_major=False)
Hd = tf.nn.dropout(Hr, keep_prob)
Hf = tf.reshape(Hd, [-1, CELLSIZE])
Ylogits = tf.contrib.layers.linear(Hf, NCLASSES)
Y = tf.nn.softmax(Ylogits)
Yp = tf.argmax(tf.slice(Y[0], [0], [4]), 0)
Yd = tf.argmax(tf.slice(Y[0], [4], [52]), 0)

if traindatafile : train_classifier(numpypath, ckptpath, lr, iters)

if testfile : parsing(testfile,wordvecpath, numpypath, ckptpath, Yp, Yd, H, X, XL, Hin, keep_prob)



logging.info('end of program')
