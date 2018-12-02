import os
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

train_sentence, train_intent = load_train_data(src_train='/data/atis.train.w-intent.iob',
                                               src_stop_word='/stopwords/english')

# test_sentence, test_intent = load_test_data(src_test='/data/atis.test.w-intent.iob',
#                                             src_stop_word='/stopwords/english')

# train_sentence, train_intent = load_train_data(src_train='/data/aaa.iob',
#                                                src_stop_word='/stopwords/english')
#
# test_sentence, test_intent = load_test_data(src_test='/data/bbb.iob',
#                                             src_stop_word='/stopwords/english')


w2v_dict = generator_dict(train_sentence)
train_sentence_encoder = encoder(train_sentence, w2v_dict)

# n_estimators = 30
# params_type = np.arange(10, 300, 10)
# param_test1 = {'n_estimators': params_type}

# max_depth = 100
params_type = np.arange(1, 3, 1)
param_test1 = {'criterion': ['gini','entropy']}

RFC = RandomForestClassifier(criterion='gini', random_state=1, n_jobs=1, n_estimators=30,
                             max_depth=100, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=600,
                             min_weight_fraction_leaf=0)

GSCV = GridSearchCV(estimator=RFC, param_grid=param_test1, cv=5, n_jobs=8)
GSCV.fit(train_sentence_encoder, train_intent)

fig = plt.figure(1, figsize=(16, 12))
plt.clf()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
scores = GSCV.cv_results_
accuracy = scores['mean_test_score']
std = scores['std_test_score']

ax1.plot(params_type, accuracy, linewidth=2)
ax2.plot(params_type, std, linewidth=2)
plt.axis('tight')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('accuracy')
ax2.set_xlabel('n_estimators')
ax2.set_ylabel('std_accuracy')
plt.show()
print(accuracy)
print(std)
