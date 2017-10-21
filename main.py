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

for i in range(len(sentence)):
  str = sentence[i].split()
  sentence[i] = [word for word in str if(word not in stop_word_file)]
  sentence[i] = ' '.join(sentence[i])

vectorizer = CountVectorizer()
vectorizer.fit(sentence)
vector = vectorizer.transform(sentence)
end = vector.toarray()
