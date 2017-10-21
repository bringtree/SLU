import os
from sklearn.feature_extraction.text import CountVectorizer
import util
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

pwd = os.getcwd()

# load atis corpora
src_train = pwd + '/data' + '/atis.train.w-intent.iob'
train_file = util.load(src_train).split()
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
src_stop_word = pwd + '/stopwords' + '/english'
stop_word_file = util.load(src_stop_word).split('\n')

# filter stopping word.
for i in range(len(train_sentence) - 1):
  str = train_sentence[i].split()
  train_sentence[i] = [word for word in str if (word not in stop_word_file)]
  train_sentence[i] = ' '.join(train_sentence[i])

# load intent corpora
src_train = pwd + '/data' + '/atis.train.w-intent.iob'
train_intent_file = util.load(src_train).split('\n')
for i in range(len(train_intent_file) - 1):
  train_intent_file[i] = train_intent_file[i].split()
  train_intent_file[i] = train_intent_file[i][-1]
del train_intent_file[len(train_intent_file) - 1]

# encode train_sentence to word vector
vectorizer = CountVectorizer()
vectorizer.fit(train_sentence)
vector = vectorizer.transform(train_sentence)
train_input_x = vector.toarray()
# encode train_intent_file tot intent vector
vectorizer = CountVectorizer()
vectorizer.fit(train_intent_file)
vector = vectorizer.transform(train_intent_file)
train_output_y = vector.toarray()

# load atis corpora
src_test = pwd + '/data' + '/atis.test.w-intent.iob'
test_file = util.load(src_test).split()
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
src_stop_word = pwd + '/stopwords' + '/english'
stop_word_file = util.load(src_stop_word).split('\n')

# filter stopping word.
for i in range(len(test_sentence) - 1):
  str = test_sentence[i].split()
  test_sentence[i] = [word for word in str if (word not in stop_word_file)]
  test_sentence[i] = ' '.join(test_sentence[i])

# load intent corpora
src_test = pwd + '/data' + '/atis.test.w-intent.iob'
test_intent_file = util.load(src_test).split('\n')
for i in range(len(test_intent_file) - 1):
  test_intent_file[i] = test_intent_file[i].split()
  test_intent_file[i] = test_intent_file[i][-1]
del test_intent_file[len(test_intent_file) - 1]

# encode test_sentence to word vector
vectorizer = CountVectorizer()
vectorizer.fit(test_sentence)
vector = vectorizer.transform(test_sentence)
test_x = vector.toarray()
# encode test_intent_file tot intent vector
vectorizer = CountVectorizer()
vectorizer.fit(test_intent_file)
vector = vectorizer.transform(test_intent_file)
test_y = vector.toarray()

clf = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=1, n_jobs=2)
clf = clf.fit(train_input_x, train_output_y)
scores = cross_val_score(clf,train_input_x,train_output_y)
print(scores.mean())

scores = cross_val_score(clf,test_x,test_y)
print(scores.mean())
