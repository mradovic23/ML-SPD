import os
import logging
import argparse
import pandas as pd
from sklearn import model_selection
from sklearn import svm
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

def get_corpus(path):
    '''
    Given root folder name, function goes through all subfolders,
    reads files and their positive/negative ratings and
    puts all informations in Data Frame structure.
    param input: corpus root folder name
    return: Data Frame structure with extracted informations from corpus
    '''
    corpus, classes = [], []
    for root, directories, files in os.walk(path):
        for file in files:
            absolute_path = os.path.join(root, file)
            if os.path.dirname(absolute_path).endswith('pos'):
                classes.append('POSITIVE')
            elif os.path.dirname(absolute_path).endswith('neg'):
                classes.append('NEGATIVE')
            else:
                logging.ERROR('Unknown type of class')
            f = open(absolute_path, 'r')
            corpus.append(f.read())

    data = {'Text': corpus, 'Rating': classes}
    df = pd.DataFrame(data)

    return df

def bag_of_words(corpus):
    '''
    Given corpus, function creates a vector of token counts using Count Vectorizer class.
    param input: corpus
    return: bag of words model - vector of token counts

    '''
    vectorizer = CountVectorizer()
    c = vectorizer.fit_transform(corpus)

    return c.toarray()

def svm_classifier(X_train, X_test, y_train, y_test):
    '''
    Given dataset for training and testing, functions creates a Support Vector Machine classifier,
    trains the model using the training set, predicts the response for the test dataset
    and returns the score between predicted and test dataset.
    param input: dataset for training and testing
    return: score between predicted and test dataset

    '''
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)

def nb_classifier(X_train, X_test, y_train, y_test):
    '''
    Given dataset for training and testing, functions creates a Naive Bayes classifier,
    trains the model using the training set, predicts the response for the test dataset
    and returns the score between predicted and test dataset.
    param input: dataset for training and testing
    return: score between predicted and test dataset

    '''
    clf = naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)

if __name__ == '__main__':

    # Set the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-l', '--log_level', required=False, help='Set logging level [CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET]')
    ap.add_argument('-t', '--test_percentage', required=False, help='Set the percentage for test data [0-100]')
    args = vars(ap.parse_args())

    # Set logging level
    log_level = args['log_level'] if args['log_level'] != None else 'INFO'
    numeric_level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(level=numeric_level)

    # Get the dataset for Serbian reviews
    file_path_srb = 'SerbMR-3C.csv'
    try:
        corpus_srb = pd.read_csv(file_path_srb, encoding='utf-8')
    except OSError:
        logging.ERROR('Could not open: {}', file_path_srb)
        sys.exit()
    corpus_srb.columns = ['Text', 'Rating']
    corpus_srb, classes_srb = corpus_srb.Text, corpus_srb.Rating


    # Get the dataset for English reviews
    file_path_eng = 'review_polarity/txt_sentoken'
    try:
        corpus = get_corpus(file_path_eng)
    except OSError:
        logging.ERROR('Could not open: {}', file_path_eng)
        sys.exit()
    corpus_eng, classes_eng = corpus.Text, corpus.Rating

    # Get the bag of words model
    bag_of_words_model_srb = bag_of_words(corpus_srb)
    logging.debug('Bag of words for Serbian reviews:\n {}'.format(bag_of_words_model_srb))
    bag_of_words_model_eng = bag_of_words(corpus_eng)
    logging.debug('Bag of words for English reviews:\n {}'.format(bag_of_words_model_eng))
    
    # Split the dataset for training and testing
    test_size = int(args['test_percentage']) * 0.01 if args['test_percentage'] != None else 0.3

    X_train_srb, X_test_srb, y_train_srb, y_test_srb = model_selection.train_test_split(
                                        bag_of_words_model_srb, classes_srb, test_size=test_size)

    X_train_eng, X_test_eng, y_train_eng, y_test_eng = model_selection.train_test_split(
                                        bag_of_words_model_eng, classes_eng, test_size=test_size)

    # SVM algorithm
    score = svm_classifier(X_train_srb, X_test_srb, y_train_srb, y_test_srb)
    logging.info('SVM accuracy for Serbian reviews: {}'.format(score))
    score = svm_classifier(X_train_eng, X_test_eng, y_train_eng, y_test_eng)
    logging.info('SVM accuracy for English reviews: {}'.format(score))

    # NB algorithm
    score = nb_classifier(X_train_srb, X_test_srb, y_train_srb, y_test_srb)
    logging.info('NB accuracy for Serbian reviews: {}'.format(score))
    score = nb_classifier(X_train_eng, X_test_eng, y_train_eng, y_test_eng)
    logging.info('NB accuracy for English reviews: {}'.format(score))
