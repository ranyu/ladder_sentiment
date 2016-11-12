import random
import input_news
import ladder_network
import tensorflow as tf

import numpy as np

news_set = input_news.read_data_sets()
print news_set.train_unlabeled.num_examples, "unlabeled training examples"
print news_set.train_labeled.num_examples, "labeled training examples"
print news_set.validation.num_examples, "validation examples"
print news_set.test.num_examples, "test examples"


hyperparameters = {
  "learning_rate": 0.01,
  "noise_level": 0.2,
  "input_layer_size": 100,
  "class_count": 3,
  "encoder_layer_definitions": [
    (100, tf.nn.relu), # first hidden layer
    (50, tf.nn.relu),
    (3, tf.nn.softmax) # output layer
  ],
  "denoising_cost_multipliers": [
    1000, # input layer
    0.5, # first hidden layer
    0.1,
    0.1 # output layer
  ]
}

graph = ladder_network.Graph(**hyperparameters)
#limit the usage of GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
with ladder_network.Session(graph,config=config) as session:
  for step in xrange(100000):
    if step % 5 == 0:
      news, labels = news_set.train_labeled.next_batch(100)
      session.train_supervised_batch(news, labels, step)
    else:
      news, _ = news_set.train_unlabeled.next_batch(100)
      session.train_unsupervised_batch(news, step)

    if step % 200 == 0:
      save_path = session.save()
      accuracy = session.test(
        news_set.validation.news, news_set.validation.labels, step)
      print
      print "Model saved in file: %s" % save_path
      print "Accuracy: %f" % accuracy
