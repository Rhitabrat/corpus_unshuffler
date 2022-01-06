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
    # path: str

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

    def __init__(self, path: str, sentence_type: int=1):
        """ Constructor for Sentence Classification """
        # df = self.load_data(path=path, sentence_type=sentence_type)
        df = self.load_data(path=path)

    def load_data(self, path: str):
        data = []

        files = [os.path.join(path, f) for f in os.listdir(path)]
        files = natsorted(files)

        for f in files:
            with open(f, "r", encoding="unicode_escape") as myfile:
                article = myfile.read()
                # doc = article.split("\n\n")
                data.append(article)

        df = pd.DataFrame(data, columns=["article"] )

        print(f"df: \n{df}")
        print(f"df.article: \n{df.article}")

        list_of_sentences = []
        list_of_labels = []

        ''' 
        LABELS 
        opening sentences: a -> 1
        closing sentences: b -> 2
        other   sentences: c -> 3
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

        selector = df_labeled['label'] == 1
        df_labeled[selector][['sentence']].to_csv("headings.csv", index=False)
        selector = df_labeled['label'] == 2
        df_labeled[selector][['sentence']].to_csv("closings.csv", index=False)
        selector = df_labeled['label'] == 3
        df_labeled[selector][['sentence']].to_csv("others.csv", index=False)

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

    def load_data_new(self, path: str, sentence_type: int):
        '''
        LABELS
        opening sentences: a -> 1
        closing sentences: b -> 2
        other   sentences: c -> 3
        '''

        label = [1, 2, 3]

        sentences = []
        labels = []

        with open(path, "r", encoding="unicode_escape") as file:
            sentences = file.read().splitlines()
            file.close()

        labels = [sentence_type] * len(sentences)

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, stratify=df_labeled['label'])

        print(f"sentences:\n{sentences}")
        print(f"\n\nlabels:\n{labels}")

        print(f"sentence count: {len(sentences)}")
        print(f"labels count: {len(labels)}")

    def build_pipeline(self) -> None:

        parameters = \
            {
                # 'count__max_df': (0.5, 0.75, 1.0),
                # 'count__max_features': (None, 5000, 10000, 50000),
                # 'count__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
                # 'tfidf__use_idf': (True, False),
                # 'tfidf__norm': ('l1', 'l2'),
                # 'tfidf__max_features': [100, 2000],
                # 'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
                'tfidf__use_idf': (True, False),
                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)],
                'tfidf__stop_words': [None, 'english'],

                # 'clf__max_iter': (20,),
                # 'clf__alpha': (0.00001, 0.000001),
                # 'clf__penalty': ('l2', 'elasticnet'),
                # 'clf__max_iter': (10, 50, 80),
                'clf__alpha': (1, 0.1, 0.01)
            }

        pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=10)


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

# path = 'data/bbcsport/athletics/'
# path = 'data/bbcsport/cricket/'
# path = 'data/bbcsport/football/'
# path = 'data/bbcsport/rugby/'
# path = 'data/bbcsport/tennis/'
path = 'data/bbcsport/'
sc = SentenceClassifier(path=path)

# path = "data/huff_headlines_clean.txt"
# sc = SentenceClassifier(path=path, sentence_type=1)
# sc.train()
# sc.test()
