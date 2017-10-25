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

clf = RandomForestClassifier(criterion='gini', random_state=1, n_jobs=-1, n_estimators=30,
                             max_depth=100, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=600,
                             min_weight_fraction_leaf=0)

train_socre = 0

dict = generator_dict(train_sentence)

train_sentence_encoder = encoder(train_sentence,dict)

test_sentence_encoder = encoder(test_sentence, dict)

clf.fit(train_sentence_encoder, train_intent)

test_predict_intent = clf.predict(test_sentence_encoder)

with open(pwd + '/false.txt', 'w')  as file:
  str = [v + '\n' for v in test_predict_intent]
  file.writelines(str)

true_intent = load_ready_intent('/ready.txt')
print((test_intent == test_predict_intent).mean())
