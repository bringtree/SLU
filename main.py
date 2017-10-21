import os
from sklearn.feature_extraction.text import CountVectorizer

pwd = os.getcwd()
src_train = pwd + '/data' + '/atis.test.w-intent.iob'
temp_file = open(src_train, "r", encoding='utf-8')
train_file = temp_file.read()
temp_file.close()

train_file = train_file.split()
cleaned_list = []
a = ''

for x in train_file:
  if x != 'BOS' and x != 'EOS':
    a = a + x + ' '
  else:
    cleaned_list.append(a)
    a = ''

cleaned_list.append(a)
del cleaned_list[0]

sentence = []
tags = []
l = 0
m = 1
for i in range(4978):
  tags.append(cleaned_list[m])
  sentence.append(cleaned_list[l])
  l = l + 2
  m = m + 2

vectorizer = CountVectorizer()
vectorizer.fit(sentence)
vector = vectorizer.transform(sentence)
end = vector.toarray()
