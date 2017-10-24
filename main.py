import os
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import numpy as np

train_sentence, train_intent = load_train_data(src_train='/data/atis.train.w-intent.iob',
                                               src_stop_word='/stopwords/english')

# test_sentence, test_intent = load_test_data(src_test='/data/atis.test.w-intent.iob',
#                                             src_stop_word='/stopwords/english')

# train_sentence, train_intent = load_train_data(src_train='/data/aaa.iob',
#                                                src_stop_word='/stopwords/english')
#
# test_sentence, test_intent = load_test_data(src_test='/data/bbb.iob',
#                                             src_stop_word='/stopwords/english')



max_train = 0

for i in range(10, 1010, 10):
  for j in range(5, 100, 5):
    clf = RandomForestClassifier(criterion='entropy', n_estimators=i, random_state=1, n_jobs=-1,
                                 max_depth=j)

    train_socre = 0

    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(train_sentence):
      split_train_sentence, split_test_sentence = train_sentence[train_index], train_sentence[test_index]
      split_train_intent, split_test_intent = train_intent[train_index], train_intent[test_index]

      train_sentence_encoder, train_sentence_dict, = train_encoder(split_train_sentence)
      test_sentence_encoder = test_encoder(split_test_sentence, train_sentence_dict)

      clf.fit(train_sentence_encoder, split_train_intent)

      test_predict_intent = clf.predict(test_sentence_encoder)

      train_socre += (split_test_intent == test_predict_intent).mean()

    train_score_mean = train_socre / 5

    if (train_score_mean > max_train):
      max_train = train_score_mean
      print('train_score:' + str(max_train) + ' n_estimators:' + str(i) + ' max_depth:' + str(j) + '\n')
