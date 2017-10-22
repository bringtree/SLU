import os
from sklearn.feature_extraction.text import CountVectorizer
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

train_sentence, train_intent = load_train_data(src_train='/data/atis.train.w-intent.iob',
                                               src_stop_word='/stopwords/english')
# encode train_sentence to word vector
train_input_x_dict = CountVectorizer()
train_input_x_dict.fit(train_intent)
vector = train_input_x_dict.transform(train_intent)
train_input_x = vector.toarray()
# encode train_intent_file tot intent vector
train_input_y_dict = CountVectorizer()
train_input_y_dict.fit(train_intent)
vector = train_input_y_dict.transform(train_intent)
train_output_y = vector.toarray()


test_sentence, test_intent = load_test_data(src_test='/data/atis.test.w-intent.iob', src_stop_word='/stopwords/english')
# encode test_sentence to word vector
vector = train_input_x_dict.transform(test_sentence)
test_x = vector.toarray()
# encode test_intent_file tot intent vector
vector = train_input_y_dict.transform(test_intent)
test_y = vector.toarray()


clf = RandomForestClassifier(criterion='entropy', n_estimators=1, random_state=1, n_jobs=-1)
clf = clf.fit(train_input_x, train_output_y)
scores = cross_val_score(clf, train_input_x, train_output_y)
print(scores.mean())

# scores = cross_val_score(clf,test_x,test_y)
# print(scores.mean())
