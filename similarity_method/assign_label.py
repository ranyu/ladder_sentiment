#!/usr/bin/env python
# coding=utf-8

import numpy as np
from scipy.spatial.distance import cosine

def label_to_onehot(label):
    max_label = max(label)
    onehot_label = np.zeros((len(label),max_label+1),dtype=np.float)
    for num,ele in enumerate(label):
        onehot_label[num][label[num]] = 1.0
    return onehot_label

def unlabel_to_onehot(unlabel):
    onehot_unlabel = np.zeros((len(unlabel),3),dtype=np.float)
    return onehot_unlabel
    
def load_label():
    train_label = np.load('../../method/doclabel.npy')
    train_unlabel = np.load('../../method/doc_unlabel_vector.npy')
    train_label = label_to_onehot(train_label)
    train_unlabel = unlabel_to_onehot(train_unlabel)
    return train_label[:1327],train_label[1327:],train_unlabel

def load_traindata():
    train_news = np.load('../../method/docvector.npy')
    train_unlabelnews = np.load('../../method/doc_unlabel_vector.npy')
    return train_news[:1327],train_news[1327:],train_unlabelnews

train_news,test_news,train_unlabel_news = load_traindata()
train_label,test_label,train_unlabel = load_label()

for num,ele in enumerate(train_unlabel_news):
    if num % 10000 == 0:
        print num
    array = []
    for content in train_news:
        array.append(cosine(ele,content))
    train_unlabel[num] = train_label[np.argmax(np.array(array))]

np.save('../../method/doc_unlabel_vector_assign_label.npy',train_unlabel)
