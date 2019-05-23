import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import sentimentAnalysis
from sentimentAnalysis import generate_dependency_wordbag
from sentimentAnalysis import concact_aspect_term
from sentimentAnalysis import generate_weightage_window

nltk.download('stopwords')
from nltk.corpus import stopwords

df = pd.DataFrame()
df = df.append(pd.read_csv('/Users/sharandec7/PycharmProjects/project2/NaiveByes/data/dara 1_train.csv', skiprows=1,
                           names=['example_id', 'text', 'aspect_term', 'term_location', 'res']))
# df = df.append(pd.read_csv('/Users/sharandec7/PycharmProjects/project2/NaiveByes/data/data 2_train.csv', skiprows=1,
#                names=['example_id', 'text', 'aspect_term', 'term_location', 'res']))
# df
print(df.columns.values)

for index, row in df.iterrows():
    if index == 0:
        continue
    row['text'], row['aspect_term'] = concact_aspect_term(row['text'], row['aspect_term'])
    row['text'] = generate_weightage_window(row['text'], row['aspect_term'])
    # row['text'] = generate_dependency_wordbag(row['text'], row['aspect_term'])

print(sentimentAnalysis.valid_count)
print(sentimentAnalysis.invalid_count)

stopset = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}


# TFIDF vectorize
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)
Y = df.res
X = vectorizer.fit_transform(df.text)
print(X)
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
# print(kf.get_n_splits(X))
kf.get_n_splits(X)  # returns the number of splitting iterations in the cross-validator
print(kf.get_n_splits)
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

for train_index, test_index in kf.split(X):
    # print('train', train_index, 'test', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
clf = MultinomialNB()
clf.fit(X_train, y_train)

# movie_reviews_array=np.array(["jupyter ascending was a disappointing and terrible movie"])
# movie_review_vector=vectorizer.transform(movie_reviews_array)
# print(clf.predict(movie_review_vector))
# clf.predict(movie_review_vector)
predicted = clf.predict(X_test)
print(classification_report(y_test, predicted))

clf2 = svm.SVC(decision_function_shape='ovo')
clf2.fit(X, Y)
clf2 = svm.LinearSVC()
clf2.fit(X_train, y_train)

predicted = clf2.predict(X_test)
print(classification_report(y_test, predicted))
