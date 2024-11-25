import collections
import string
import time

import numpy as np
import sklearn
import sklearn.feature_extraction
import sklearn.svm
import sklearn.metrics
import gzip
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


def preprocess(text: str):
    """ Normalizes case and handles punctuation

    args:
        text: str -- raw text

    Outputs:
        list(str): tokenized text
    """

    text = text.lower()
    text = re.sub(r"http://t\.co/\w+", "", text)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()

    return tokens


def read_data():
    """Reads the dataset from the tweets_train.csv.gz and tweets_test.csv.gz files

    return : Tuple (data_train, data_test)
        data_train : List[Tuple[is_republican, tokenized_tweet]]
            is_republican : bool -- True if tweet is from a republican
            tokenized_tweet : List[str] -- the tweet, tokenized by preprocess()
        data_test : List[Tuple[None, tokenized_tweet]
            None: the Python constant "None"
            tokenized_tweet : List[str] -- the tweet, tokenized by preprocess()
    """
    republican_accounts = {"realDonaldTrump", "mike_pence", "GOP"}
    democratic_accounts = {"HillaryClinton", "timkaine", "TheDemocrats"}

    train_df = pd.read_csv("tweets_train.csv")
    data_train = []
    for _, row in train_df.iterrows():
        screen_name = row["screen_name"]
        is_republican = screen_name in republican_accounts
        tokenized_tweet = preprocess(row["text"])
        data_train.append((is_republican, tokenized_tweet))

    test_df = pd.read_csv("tweets_test.csv")
    data_test = []
    for _, row in test_df.iterrows():
        tokenized_tweet = preprocess(row["text"])
        data_test.append((None, tokenized_tweet))

    return data_train, data_test

start_time = time.time()
data = read_data()
end_time = time.time()
# print(f"Час виконання: {end_time - start_time:.6f} секунд")

def get_distribution(data):
    """ Calculates the word count distribution.

    args:
        data -- the training or testing data

    return : collections.Counter -- the distribution of word counts
    """
    word_counts = collections.Counter()
    for _, tokenized_tweet in data:
        word_counts.update(tokenized_tweet)

    return word_counts


res = get_distribution(data[0])
# print(res)

def create_features(train_data, test_data):
    """creates the feature matrices and label vector for the training and test sets.

    args:
        train_data, test_data : output of read_data() function

    returns: Tuple[train_features, train_labels, test_features]
        train_features : scipy.sparse.csr.csr_matrix -- TFIDF feature matrix for the training set
        train_labels : np.array[num_train] -- a numpy vector, where 1 stands for Republican and 0 stands for Democrat
        test_features : scipy.sparse.csr.csr_matrix -- TFIDF feature matrix for the test set
    """
    train_texts = [" ".join(tokens) for _, tokens in train_data]
    train_labels = np.array([1 if is_republican else 0 for is_republican, _ in train_data])

    test_texts = [" ".join(tokens) for _, tokens in test_data]

    vectorizer = TfidfVectorizer(
        preprocessor=lambda x: x,
        tokenizer=lambda x: x,
        token_pattern=None,
        min_df=5,
        max_df=0.4
    )
    train_features = vectorizer.fit_transform(train_texts)
    test_features = vectorizer.transform(test_texts)

    return train_features, train_labels, test_features

create_features= create_features(data[0], data[1])
# print(create_features)

def train_classifier(features, labels, C):
    """learns a classifier from the input features and labels using a specified kernel function

    args:
        features: scipy.sparse.csr.csr_matrix -- sparse matrix of features
        labels : numpy.ndarray(bool): binary vector of class labels
        C : float -- C regularization parameters

    returns: sklearn.svm.LinearSVC -- classifier trained on data
    """
    classifier = LinearSVC(
        C=C,
        loss="hinge",
        random_state=0,
        max_iter=10000
    )
    classifier.fit(features, labels)

    return classifier

train_classifier = train_classifier(create_features[0], create_features[1], 1.0)
# print(train_classifier)


def evaluate_classifier(features, labels, C=(0.01, 0.1, 1.0, 10., 100.), train_length=10000):
    """ Train multiple classifier based on the first train_length features of features/labels,
        one for each regularization parameter supplied in C, and return train/validation f1
        scores for each of the classifiers

    args:
        features: scipy.sparse.csr.csr_matrix -- sparse matrix of features
        labels : numpy.ndarray(bool): binary vector of class labels
        C : Tuple[float] -- tuple of C regularization parameters
        train_length: int -- use _first_ train_length features for training (and the rest of validation)

    return : List[Tuple[float, float]] -- list of F1 scores for training/validation for each C parameter
    """
    train_features = features[:train_length]
    train_labels = labels[:train_length]
    validation_features = features[train_length:]
    validation_labels = labels[train_length:]

    results = []
    for c in C:
        classifier = LinearSVC(C=c, loss="hinge", random_state=0, max_iter=10000)
        classifier.fit(train_features, train_labels)
        train_predictions = classifier.predict(train_features)
        validation_predictions = classifier.predict(validation_features)
        train_f1 = f1_score(train_labels, train_predictions)
        validation_f1 = f1_score(validation_labels, validation_predictions)
        results.append((train_f1, validation_f1))

    return results

evaluate_classifier = evaluate_classifier(create_features[0], create_features[1])
print(evaluate_classifier)
