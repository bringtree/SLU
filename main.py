import os
from util import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

train_sentence, train_intent = load_train_data(src_train='/data/atis.train.w-intent.iob',
                                               src_stop_word='/stopwords/english')

test_sentence, test_intent = load_test_data(src_test='/data/atis.test.w-intent.iob',
                                            src_stop_word='/stopwords/english')

split_train_sentence, split_test_sentence, split_train_intent, split_test_intent = train_test_split(train_sentence,
                                                                                                    train_intent,
                                                                                                    test_size=0.18,
                                                                                                    random_state=0)
train_input_x, train_output_y, train_input_x_dict, train_input_y_dict = train_encoder(split_train_sentence,
                                                                                      split_train_intent)
test_x, test_y = test_encoder(split_test_sentence, split_test_intent, train_input_x_dict, train_input_y_dict)

max_train = 0
max_test = 0
for i in range(100):
  for j in range(50):
    clf = RandomForestClassifier(criterion='entropy', n_estimators=(i + 1) * 10, random_state=1, n_jobs=-1,
                                 min_samples_leaf=(j + 1) * 10)

    split_train_sentence, split_test_sentence, split_train_intent, split_test_intent = train_test_split(train_sentence,
                                                                                                        train_intent,
                                                                                                        test_size=0.18,
                                                                                                        random_state=0)
    train_socre = 0

    for i in range(5):
      train_input_x, train_output_y, train_input_x_dict, train_input_y_dict = train_encoder(split_train_sentence,
                                                                                            split_train_intent)
      test_x, test_y = test_encoder(split_test_sentence, split_test_intent, train_input_x_dict, train_input_y_dict)

      clf.fit(train_input_x, train_output_y)

      train_predict = clf.predict(test_x)
      train_socre += (test_y == train_predict).mean()

    train_score_mean = train_socre / 5

    if (train_score_mean > max_train):
      max_train = train_score_mean
      print('train_score:' + str(max_train) + ' n_estimators:' + str((i + 1) * 10) + ' min_samples_leaf:' + str(
        (j + 1) * 10) + '\n')


      # test_predict = clf.predict(test_x)
      # test_score = (test_y == test_predict).mean()
      # if (test_score > max_test):
      #   max_test = test_score
      #   print('test_score:' + str(max_test) + ' n_estimators:' + str((i + 1) * 10) + ' min_samples_leaf:' + str(
      #     (j + 1) * 10) + '\n')



# clf = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=1, n_jobs=-1, min_samples_leaf=100)
# clf.fit(train_input_x, train_output_y)
# scores = cross_val_score(clf, train_input_x, train_output_y)
# print(scores.mean())

# predict = clf.predict(train_input_x)
# print((train_output_y == predict).mean())
#
# predict = clf.predict(test_x)
# print((test_y == predict).mean())

# scores = cross_val_score(clf, train_input_x, train_output_y)

# print(scores.mean())
