import pandas as pd
import os
from natsort import natsorted
import nltk
nltk.download('punkt')


def load_data(path):
    data = []

    files = [os.path.join(path, f) for f in os.listdir(path)]
    files = natsorted(files)

    for f in files:
        with open(f, "r", encoding="unicode_escape") as myfile:
            article = myfile.read()
            # doc = article.split("\n\n")
            data.append(article)

    df = pd.DataFrame(data, columns=["article"] )
    return df


# path = '/content/drive/Othercomputers/My MacBook Pro/PSU/NLP Lab/Steve-Thesis/Data/bbcsport/football/'
path = 'data/bbcsport/football/'
df = load_data(path)
print(f"df: \n{df}")
print(f"df.article: \n{df.article}")

list_of_sentences = []
list_of_labels = []

''' 
LABELS 

opening sentence, a -> 1
closing sentence, b -> 2
none, c -> 3
'''

label = [1, 2, 3]

for each_article in df.article:
    list_of_labels.append(label[0])
    num_of_mid_sentences = 0
    doc = each_article.split("\n\n")

    for each_paragraph in doc:
        sentences = nltk.sent_tokenize(each_paragraph)
        for each_sentence in sentences:
            list_of_sentences.append(each_sentence)
            num_of_mid_sentences += 1

    for i in range(num_of_mid_sentences - 2):
        list_of_labels.append(label[2])

    list_of_labels.append(label[1])

df_labeled = pd.DataFrame(list(zip(list_of_sentences, list_of_labels)), columns=["sentence", "label"])
print(f"df_labeled: {df_labeled}")

from sklearn.model_selection import train_test_split
X = df_labeled['sentence']
y = df_labeled['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=df_labeled['label'])
print(f"y_train Counts: {y_train.value_counts()}")
print(f"y_test Counts: {y_test.value_counts()}")

# Logistic Regression
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# test_counts = X_test.value_counts()
score = classifier.score(X_test, y_test)

# print(f"Test Result Counts: {test_counts}")
pd.set_option("display.max_rows", None, "display.max_columns", None)
print("Accuracy:", score)
# print(f"X_test: {X_test}")
# print(f"y_test: {y_test}")

from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")
print(f"Accuracy Per Label: {cm.diagonal()/cm.sum(axis=1)}")

