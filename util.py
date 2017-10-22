import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def load(src):
  temp_file = open(src, "r", encoding='utf-8')
  save_variable = temp_file.read()
  temp_file.close()
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

  # str = [v+'\n' for v in test_intent]
  # file = open(pwd + '/ready.txt', 'w')
  # file.writelines(str)
  # file.close()
  return test_sentence, test_intent


def train_encoder(train_sentence, train_intent):
  # encode train_sentence to word vector
  train_input_x_dict = CountVectorizer()
  train_input_x_dict.fit(train_sentence)
  vector = train_input_x_dict.transform(train_sentence)
  train_input_x = vector.toarray()

  # encode train_intent_file tot intent vector
  train_input_y_dict = CountVectorizer()
  train_input_y_dict.fit(train_intent)
  vector2 = train_input_y_dict.transform(train_intent)
  train_output_y = vector2.toarray()

  return train_input_x, train_output_y, train_input_x_dict, train_input_y_dict


def test_encoder(test_sentence, test_intent, train_input_x_dict, train_input_y_dict):
  # encode test_sentence to word vector
  vector = train_input_x_dict.transform(test_sentence)
  test_x = vector.toarray()

  # encode test_intent_file tot intent vector
  vector2 = train_input_y_dict.transform(test_intent)
  test_y = vector2.toarray()
  return test_x, test_y
