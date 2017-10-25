import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def load(src):
  with open(src, "r", encoding='utf-8') as file:
    save_variable = file.read()
  return save_variable


def load_train_data(src_train, src_stop_word='/stopwords/english', workspace=None):
  if workspace == None:
    pwd = os.getcwd()
  else:
    pwd = workspace

  # load atis corpora
  src = pwd + src_train
  train_file = load(src).split()
  train_sentence = []
  str = ''
  for word in train_file:
    if word == 'BOS':
      str = ''
    elif word == 'EOS':
      train_sentence.append(str)
    else:
      str = str + word + ' '

  del str

  # load stop_word copora
  src_stop_word = pwd + src_stop_word
  stop_word_file = load(src_stop_word).split('\n')

  # filter stopping word.
  for i in range(len(train_sentence) - 1):
    str = train_sentence[i].split()
    train_sentence[i] = [word for word in str if (word not in stop_word_file)]
    train_sentence[i] = ' '.join(train_sentence[i])

  # load intent corpora
  src = pwd + src_train
  train_intent = load(src).split('\n')
  for i in range(len(train_intent) - 1):
    train_intent[i] = train_intent[i].split()
    train_intent[i] = train_intent[i][-1]
  del train_intent[len(train_intent) - 1]
  return np.array(train_sentence), np.array(train_intent)


def load_test_data(src_test, src_stop_word='/stopwords/english', workspace=None):
  if workspace == None:
    pwd = os.getcwd()
  else:
    pwd = workspace
  # load atis corpora
  src = pwd + src_test
  test_file = load(src).split()
  test_sentence = []
  str = ''
  for word in test_file:
    if word == 'BOS':
      str = ''
    elif word == 'EOS':
      test_sentence.append(str)
    else:
      str = str + word + ' '

  del str

  # load stop_word copora
  src_stop_word = pwd + src_stop_word
  stop_word_file = load(src_stop_word).split('\n')

  # filter stopping word.
  for i in range(len(test_sentence) - 1):
    str = test_sentence[i].split()
    test_sentence[i] = [word for word in str if (word not in stop_word_file)]
    test_sentence[i] = ' '.join(test_sentence[i])

  # load intent corpora
  src = pwd + src_test
  test_intent = load(src).split('\n')
  for i in range(len(test_intent) - 1):
    test_intent[i] = test_intent[i].split()
    test_intent[i] = test_intent[i][-1]
  del test_intent[len(test_intent) - 1]

  # generator the ready data file: ready.txt

  # str = [v+'\n' for v in test_intent]
  # file = open(pwd + '/ready.txt', 'w')
  # file.writelines(str)
  # file.close()
  return test_sentence, test_intent


def generator_dict(word):
  dict = CountVectorizer()
  dict.fit(word)
  return dict


def encoder(row, dict):
  # encode test_sentence to word vector
  vector = dict.transform(row)
  test_x = vector.toarray()

  return test_x


def load_ready_intent(src, workspace=None):
  if workspace == None:
    pwd = os.getcwd()
  else:
    pwd = workspace

  test_file = load(pwd + src).split()
  test_intent = []
  str = ''
  for word in test_file:
    if word == 'BOS':
      str = ''
    elif word == 'EOS':
      test_intent.append(str)
    else:
      str = str + word + ' '

  del str

  test_intent = [v.split()[-1] for v in test_file if v != '']
  return test_intent
