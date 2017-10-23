import os
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import numpy as np

train_sentence, train_intent = load_train_data(src_train='/data/atis.train.w-intent.iob',
                                               src_stop_word='/stopwords/english')

test_sentence, test_intent = load_test_data(src_test='/data/atis.test.w-intent.iob',
                                            src_stop_word='/stopwords/english')

# train_sentence, train_intent = load_train_data(src_train='/data/aaa.iob',
#                                                src_stop_word='/stopwords/english')
#
# test_sentence, test_intent = load_test_data(src_test='/data/bbb.iob',
#                                             src_stop_word='/stopwords/english')
pwd = os.getcwd()

clf = RandomForestClassifier(criterion='entropy', n_estimators=40, random_state=1, n_jobs=-1,
                             max_depth=80)

train_socre = 0

train_sentence_encoder, train_intent_encoder, train_sentence_dict, train_intent_dict = train_encoder(test_sentence,
                                                                                                     test_intent)

test_sentence_encoder = test_encoder(test_sentence, train_sentence_dict)

clf.fit(train_sentence_encoder, train_intent_encoder)

test_predict_intent_encoder = clf.predict(test_sentence_encoder)

test_predict_intent = train_intent_dict.inverse_transform(test_predict_intent_encoder)

str = []
for v in test_predict_intent:
  if v.size == 0:
    str.append('\n')
  else:
    str.append(v[0]+'\n')

file = open(pwd + '/false.txt', 'w')
file.writelines(str)
file.close()

# train_socre += ((test_intent == test_predict_intent).mean())

true_intent = load_ready_intent('/ready.txt')
k = 0
for i in range(0, len(true_intent)):
  if test_predict_intent[i].size != 0:
    if true_intent[i] == test_predict_intent[i][0]:
      k = k + 1
print(float(k) / len(true_intent))
