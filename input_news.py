# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np

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
    train_label = np.load('../../../method/doclabel.npy')
    train_unlabel = np.load('../../../method/doc_unlabel_vector.npy')
    train_label = label_to_onehot(train_label)
    train_unlabel = unlabel_to_onehot(train_unlabel)
    return train_label[:1327],train_label[1327:],train_unlabel

def load_traindata():
    train_news = np.load('../../../method/docvector.npy')
    train_unlabelnews = np.load('../../../method/doc_unlabel_vector.npy')
    return train_news[:1327],train_news[1327:],train_unlabelnews

train_news,test_news,train_unlabel_news = load_traindata()
train_label,test_label,train_unlabel = load_label()

class DataSet(object):

  def __init__(self, news, labels, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      self._num_examples = news.shape[0]

    self._news = news
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def news(self):
    return self._news

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_new = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_new for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._news = self._news[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._news[start:end], self._labels[start:end]


def read_data_sets(
    fake_data=False, one_hot=False,
    validation_size=200):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train_unlabeled = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.train_labeled = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
    return data_sets

  train_news,test_news,train_unlabel_news = load_traindata()
  train_label,test_label,train_unlabel = load_label()

  assert len(train_label) == len(train_news)
  assert len(test_label) == len(test_news)


  validation_news = train_news[:validation_size]
  validation_labels = train_label[:validation_size]
  train_labeled_news = train_news[validation_size:]
  train_labeled_labels = train_label[validation_size:]

  data_sets.train_labeled = DataSet(train_labeled_news, train_labeled_labels)
  data_sets.train_unlabeled = DataSet(train_unlabel_news, train_unlabel)
  data_sets.validation = DataSet(validation_news, validation_labels)
  data_sets.test = DataSet(test_news, test_label)

  return data_sets
