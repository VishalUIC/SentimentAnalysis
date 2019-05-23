from compiler import transformer

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import nltk
from sklearn.preprocessing import StandardScaler

import sentimentAnalysis_weightage
from sentimentAnalysis_weightage import generate_dependency_wordbag
from sentimentAnalysis_weightage import concact_aspect_term
from sentimentAnalysis_weightage import generate_weightage_window
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import tree

nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.DataFrame()
df = df.append(pd.read_csv('/Users/sharandec7/PycharmProjects/project2/NaiveByes/data/dara 1_train.csv', skiprows=1,
                           names=['example_id', 'text', 'aspect_term', 'term_location', 'res']))
# df = df.append(pd.read_csv('/Users/sharandec7/PycharmProjects/project2/NaiveByes/data/data 2_train.csv', skiprows=1,
#                names=['example_id', 'text', 'aspect_term', 'term_location', 'res']))
# df
# print(df.columns.values)

df_data = pd.DataFrame(q
df_data = df_data.append(pd.read_csv('/Users/sharandec7/Downloads/ABSentimentAnalysis/Data-1_test.csv', skiprows=1,
                                     names=['example_id', 'text', 'aspect_term', 'term_location', np.zeros]))

# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
#
# example_sent = "This is a sample sentence, showing off the stop words filtration."
#
# stop_words = set(stopwords.words('english'))
#
# word_tokens = word_tokenize(example_sent)
#
# filtered_words = [word for word in word_tokens if word not in stopwords.words('english')]
#
# print(word_tokens)
# print(filtered_words)

# example_sent = "This is a sample sentence, showing off the stop words filtration."
# filtered_word_list =  example_sent[:]
# for word in example_sent: # iterate over word_list
#   if word in stopwords.words('english'):
#     filtered_word_list.remove(word)
# print(filtered_word_list)

for index, row in df.iterrows():
    if index == 0:
        continue
    row['text'], row['aspect_term'] = concact_aspect_term(row['text'], row['aspect_term'])
    df.at[index, "text"] = generate_weightage_window(row['text'], row['aspect_term'])
    # print("sharan : ----------- "+row['text'])
    # row['text'] = generate_dependency_wordbag(row['text'], row['aspect_term'])

print(sentimentAnalysis_weightage.valid_count)
print(sentimentAnalysis_weightage.invalid_count)

sentimentAnalysis_weightage.valid_count = 0
sentimentAnalysis_weightage.valid_count = 0

for index, row in df_data.iterrows():
    if index == 0:
        continue
    row['text'], row['aspect_term'] = concact_aspect_term(row['text'], row['aspect_term'])
    df_data.at[index, "text"] = generate_weightage_window(row['text'], row['aspect_term'])
    # print("sharan : ----------- "+row['text'])
    # row['text'] = generate_dependency_wordbag(row['text'], row['aspect_term'])

print(sentimentAnalysis_weightage.valid_count)
print(sentimentAnalysis_weightage.invalid_count)

stopset = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}

# TFIDF vectorize
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

# newtfidf_df  = tfidf_vectorizer.fit_transform(text11.values.flatten()).toarray()
#     tfidfvocab = tfidf_vectorizer.get_feature_names()
#     testtfidf = tfidf_vectorizer.transform(texta.values.flatten()).toarray()
# X_test = testtfidf


Y = df.res
X = vectorizer.fit_transform(df.text)

# print(X)
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
# print(kf.get_n_splits(X))

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

print("CHECK: " + str(df.columns[df.isna().any()].tolist()))
# CHECK: ['res']

X_data_test = vectorizer.transform(df_data.text)

# for i in range(10):
kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validator
print(kf.get_n_splits)
for train_index, test_index in kf.split(X):
    # print('train', train_index, 'test', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

print("\n---------------------MULTINOMIAL NB---------------------")
clf_MNB = MultinomialNB()
clf_MNB.fit(X_train, y_train)
# movie_reviews_array=np.array(["jupyter ascending was a disappointing and terrible movie"])
# movie_review_vector=vectorizer.transform(movie_reviews_array)
# print(clf.predict(movie_review_vector))
# clf.predict(movie_review_vector)
predicted = clf_MNB.predict(X_test)

# print(classification_report(y_test, predicted))
# print("ACCURACY: " + str(accuracy_score(y_test, predicted)))


# print("SVC SVM")
# clf2 = svm.SVC(decision_function_shape='ovo')
# clf2.fit(X_train, y_train)
# predicted = clf2.predict(X_test)
# print(classification_report(y_test, predicted))
# print(accuracy_score(y_test, predicted))

# scaler = StandardScaler(copy=True, with_mean=False, with_std=True)

# count_vectorizer = CountVectorizer(binary=True)
# tfidf_transformer = TfidfTransformer(use_idf=True)
# X_train = count_vectorizer.fit_transform(X_train)
# tfidf_train_data = transformer.fit_transform(X_train)
#
# X_data_test = count_vectorizer.transform(X_data_test)
# tfidf_test_data = tfidf_transformer.transform(X_data_test)


# X_train = vectorizer.fit_transform(df.text.flatten()).toarray()
# vocab = vectorizer.get_feature_names()
# X_data_test = vectorizer.transform(X_data_test.flatten()).toarray()

# X_train = scaler.fit_transform(X_train)
# clf_Lsvm.fit(X_train, y_train)
# # X_data_test = scaler.transform(X_data_test)
# predicted = clf_Lsvm.predict(X_data_test)
# print(predicted)
# print(classification_report(y_test, predicted))
# print("ACCURACY: " + str(accuracy_score(y_test, predicted)))

print("\n---------------------LINEAR SVC----------------------------")
clf_Lsvm = svm.LinearSVC()
clf_Lsvm.fit(X_train, y_train)
predicted = clf_Lsvm.predict(X_data_test)
#print(classification_report(y_test, predicted))
#print("ACCURACY: " + str(accuracy_score(y_test, predicted)))

# print(len(predicted))
# print(len(df_data['example_id']))
l = list(df_data['example_id'])
# l.pop(0)
i = 0

f = open("SaiSharan_Nagulapalli_ReenaMary_Puthota_Data-1.txt", "w+")
for p in predicted:
    print l[i] + ";;" + p
    f.write(l[i] + ";;" + p + "\r\n")
    # f.write(l[i] + "\r\n")
    i = i + 1
# print(classification_report(y_test, predicted))

print("\n---------------------LOGISTIC REGRESSION---------------------")
clf_LR = LogisticRegression()
clf_LR.fit(X_train, y_train)
predicted = clf_LR.predict(X_test)
print(classification_report(y_test, predicted))
print("ACCURACY: " + str(accuracy_score(y_test, predicted)))

print("\n---------------------MULTINOMIAL LOGISTIC REGRESSION---------------------")
clf_MLR = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')
clf_MLR.fit(X_train, y_train)
predicted = clf_MLR.predict(X_test)
print(classification_report(y_test, predicted))
print("ACCURACY: " + str(accuracy_score(y_test, predicted)))

print("\n---------------------DECISION TREE---------------------")
clf_DT = tree.DecisionTreeClassifier()
clf_DT.fit(X_train, y_train)
predicted = clf_DT.predict(X_test)
print(classification_report(y_test, predicted))
print("ACCURACY: " + str(accuracy_score(y_test, predicted)))
