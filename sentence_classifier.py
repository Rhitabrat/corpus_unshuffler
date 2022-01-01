import pandas as pd
import os
import nltk
from natsort import natsorted
# Type Hinting Libraries
from nptyping import NDArray
from typing import List, Any

nltk.download('punkt')


class SentenceClassifier():
    """
    This class object classifies sentences as either
    """

    # Properties Set During Compile Time
    sizes: NDArray[int]
    layers: int
    input_size: int
    hidden_size: int
    output_size: int

    # Properties set during Runtime
    path: str

    # Data and Labels for Training and Testing
    test_data: NDArray[Any, Any]
    test_labels: NDArray[Any]
    train_data: NDArray[Any, Any]
    train_labels: NDArray[Any]

    X_train: None
    X_test: None
    y_train: None
    y_test: None

    classifier: None

    def __init__(self, path: str):
        """ Constructor for Sentence Classification """
        df = self.load_data(path)

    def load_data(self, path):
        data = []

        files = [os.path.join(path, f) for f in os.listdir(path)]
        files = natsorted(files)

        for f in files:
            with open(f, "r", encoding="unicode_escape") as myfile:
                article = myfile.read()
                # doc = article.split("\n\n")
                data.append(article)

        df = pd.DataFrame(data, columns=["article"] )

        # return df
        # df = self.load_data(path)


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

        ########################################################################
        # Move this train_test_split to laod_data
        from sklearn.model_selection import train_test_split
        X = df_labeled['sentence']
        y = df_labeled['label']
        ########################################################################

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, stratify=df_labeled['label'])
        print(f"y_train Counts: {self.y_train.value_counts()}")
        print(f"y_test Counts: {self.y_test.value_counts()}")

    def train(self):

        # Logistic Regression
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        vectorizer.fit(self.X_train)
        self.X_train = vectorizer.transform(self.X_train)
        self.X_test  = vectorizer.transform(self.X_test)

        from sklearn.linear_model import LogisticRegression
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X_train, self.y_train)

    def test(self):
        # test_counts = X_test.value_counts()
        score = self.classifier.score(self.X_test, self.y_test)

        # print(f"Test Result Counts: {test_counts}")
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print("Accuracy:", score)
        # print(f"X_test: {X_test}")
        # print(f"y_test: {y_test}")

        from sklearn.metrics import confusion_matrix
        y_pred = self.classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"Confusion Matrix: \n{cm}")
        print(f"Accuracy Per Label: {cm.diagonal()/cm.sum(axis=1)}")


# path = '/content/drive/Othercomputers/My MacBook Pro/PSU/NLP Lab/Steve-Thesis/Data/bbcsport/football/'
path = 'data/bbcsport/football/'
sc = SentenceClassifier(path)
sc.train()
sc.test()
