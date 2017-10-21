import os
from sklearn.feature_extraction.text import CountVectorizer
import util

pwd = os.getcwd()

# load atis corpora
src_train = pwd + '/data' + '/atis.train.w-intent.iob'
train_file = util.load(src_train).split()
sentence = []
str = ''
for word in train_file:
  if word == 'BOS':
    str = ''
  elif word == 'EOS':
    sentence.append(str)
  else:
    str = str + word + ' '

del str

# load stop_word copora
src_stop_word = pwd + '/stopwords' + '/english'
stop_word_file = util.load(src_stop_word).split('\n')

# filter stopping word.
for i in range(len(sentence)-1):
  str = sentence[i].split()
  sentence[i] = [word for word in str if(word not in stop_word_file)]
  sentence[i] = ' '.join(sentence[i])

# load intent corpora
src_train = pwd + '/data' + '/atis.train.w-intent.iob'
intent_file = util.load(src_train).split('\n')
for i in range(len(intent_file)-1):
  intent_file[i] = intent_file[i].split()
  intent_file[i] = intent_file[i][-1]
del intent_file[len(intent_file)-1]



vectorizer = CountVectorizer()
vectorizer.fit(sentence)
vector = vectorizer.transform(sentence)
input_x = vector.toarray()
output_y = intent_file