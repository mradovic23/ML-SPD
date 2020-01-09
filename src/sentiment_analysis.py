# coding=UTF-8

import os
import logging
import argparse
import nltk
import string
import re
import pandas as pd
import numpy as np
import cyrtranslit
import gc
from statistics import mean
from sklearn import model_selection
from sklearn import svm
from sklearn import naive_bayes
from sklearn import neural_network
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from helper import serbian_stemmer as ss
from helper import serbian_stopwords as ssw

def get_parser():
    '''
    Function processes arguments from command line.
    param input: None
    return: None

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_level', required=False,
                        help='Set logging level [CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET]')
    parser.add_argument('-t', '--test_percentage', required=False,
                        help='Set the percentage for test data [0-100]')
    parser.add_argument('-c', '--cross_validation', required=False,
                        help='Add cross validation', action='store_true')
    parser.add_argument('-g', '--grid_search', required=False,
                        help='Add grid search for hyperparameter tuning', action='store_true')
    arguments = vars(parser.parse_args())

    return arguments

def set_logging_level(args):
    '''
    Given argument parser, function sets the logging level chosen from command line.
    param input: argument parser
    return: None

    '''
    log_level = args['log_level'] if args['log_level'] != None else 'INFO'
    numeric_level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(level=numeric_level)

def nltk_dependencies():
    '''
    Function downloads required NLTK data.
    param input: None
    return: None

    '''
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('tokenizers/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/stopwords')
    except LookupError:
        nltk.download('stopwords')

def has_cyrillic(text):
    '''
    Given text, function checks if the text has cyrillic font.
    param input: text
    return: true if text contains cyrillic letters

    '''
    return bool(re.search('[а-яА-Я]', text))

def lower(corpus):
    '''
    Given corpus, function sets all uppercase letters to lowercase letters.
    param input: corpus
    return: corpus with lowercase letters

    '''
    lower_list = []
    for c in corpus:
        lower_list.append(c.lower())

    return lower_list

def convert_to_latin(corpus):
    '''
    Given text, function converts cyrillic font to latin and replaces {š, č, ć, đ, ž} with {sx, cx, cy, dx, zx}.
    param input: corpus
    return: latin corpus cleaned from {š, č, ć, đ, ž} letters
    '''

    latin_list = []
    for c in corpus:
        if has_cyrillic(c):
            c = cyrtranslit.to_latin(c)
        c = c.lower()
        c = c.replace("š", "sx")
        c = c.replace("č", "cx")
        c = c.replace("ć", "cy")
        c = c.replace("đ", "dx")
        c = c.replace("ž", "zx")
        latin_list.append(c)

    return latin_list

def get_srb_corpus():
    '''
    Function goes through all data with Serbian reviews, reads reviews and their positive/negative/neutral ratings
    and puts all informations in Data Frame structure.
    param input: None
    return: Data Frame structure with extracted informations from corpus

    '''
    path = 'data/SerbMR-3C.csv'

    try:
        data = pd.read_csv(path, encoding='utf-8')
    except OSError:
        logging.ERROR('Could not open: {}', path)
        sys.exit()

    data.columns = ['Text', 'Rating']
    data.Text = convert_to_latin(data.Text)

    return data.Text, data.Rating

def get_eng_corpus():
    '''
    Function goes through all path subfolders with English reviews, reads files and their positive/negative ratings
    and puts all informations in Data Frame structure.
    param input: None
    return: Data Frame structure with extracted informations from corpus

    '''
    path = 'data/review_polarity/txt_sentoken'

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
            try:
                f = open(absolute_path, 'r')
                corpus.append(f.read())
            except OSError:
                logging.ERROR('Could not open: {}', absolute_path)
                sys.exit()

    data = {'Text': corpus, 'Rating': classes}
    data = pd.DataFrame(data)
    data.Text = lower(data.Text)

    return data.Text, data.Rating

def remove_punctuation(corpus, language):
    '''
    Given corpus, function removes punctuation in all corpus documents.
    param input: corpus
    return: corpus w/o punctuation

    '''
    cleaned_text = []
    replacer = str.maketrans(dict.fromkeys(string.punctuation))
    for c in corpus:
        text = c.translate(replacer)
        # Serbian language needs extra cleaning
        if language == 'Serbian':
            text = re.sub(r'[^\w\s]', '', text)
        cleaned_text.append(text)

    return cleaned_text

def remove_stopwords(corpus, language):
    '''
    Given corpus, function removes stopwords in all corpus documents for the selected language.
    param input: corpus, language
    return: corpus w/o stopwords

    '''
    if language == 'Serbian':
        stopwords = ssw.get_list_of_stopwords()
    else:
        stopwords = nltk.corpus.stopwords.words('english')

    cleaned_text = []
    for c in corpus:
        cleaned_text.append(' '.join(word for word in c.split() if word not in stopwords))

    return cleaned_text

def stemming(corpus, language):
    '''
    Given corpus and language indicator, function is doing "word normalization", ie. reducing inflection in words to their root forms
    for all words across all documents in the corpus.
    For Serbian movies, it is done by using custom serbian stemmer from: https://github.com/nikolamilosevic86/SerbianStemmer.
    For English movies, it is done by using existing stemmer from nltk - Porter Stemmer.
    param input: corpus, language indicator
    return: stemmed corpus

    '''
    stemming_list = []
    if language == 'Serbian':
        for document in corpus:
            l = ss.stem_str(document)
            stemming_list.append("".join(l))

    else:
        ps = nltk.stem.PorterStemmer()
        for document in corpus:
            words = nltk.word_tokenize(document)
            l = []
            for w in words:
                l.append(ps.stem(w))
                l.append(' ')
            stemming_list.append("".join(l))

    return stemming_list

def generate_ngrams(corpus, n):
    '''
    Given corpus and number n, function extracts all ngrams from all files in corpus.
    For n = (1, 1) this function creates a unigram model (or bag of words model).
    For n = (2, 2) this function creates a bigram model.
    For n = (1, 2) this function creates a bigram + unigram model.
    param input: corpus, number n
    return: vector of ngrams

    '''
    vectorizer = CountVectorizer(ngram_range=n, max_features=100000)
    c = vectorizer.fit_transform(corpus)

    return c.toarray()

def get_part_of_speech_words(corpus):
    '''
    Given corpus, function uses English POS tagger for tagging all words in corpus.
    param input: corpus
    return: tagged words (tuple shape (word, tag))

    '''
    tags = []
    for c in corpus:
        sentences = nltk.word_tokenize(c)
        tagged_sentences = nltk.pos_tag(sentences)
        tags.append(tagged_sentences)

    return tags

def create_vocabulary(corpus):
    '''
    Given tagged word list (tuple shape (word, tag)), function creates a list of unique sorted tuples - vocabulary.
    param input: tagged word list
    return: vocabulary

    '''
    vocabulary = []
    for c in corpus:
        vocabulary.extend(c)

    vocabulary = sorted(list(set(vocabulary)))

    return vocabulary

def create_model(corpus, vocabulary):
    '''
    Given corpus and vocabulary, functions creates a vector of token counts.
    param input: list of tagged words and tagged vocabulary
    return: appropriate (vector of token counts) model for ML algorithms

    '''
    model = np.zeros((len(corpus), len(vocabulary)))
    for i, documents in enumerate(corpus):
        document = np.zeros(len(vocabulary))
        for c in corpus[i]:
            for j, word in enumerate(vocabulary):
                if word == c:
                    document[j] += 1
        model[i] = document

    return model

def part_of_speech_tagging(corpus):
    '''
    Given corpus, function extracts tagged words, creates a vocabulary and appropriate model for ML algorithms.
    param input: corpus
    return: appropriate model for ML algorithms

    '''
    tags = get_part_of_speech_words(corpus)
    vocabulary = create_vocabulary(tags)
    pos_tag = create_model(tags, vocabulary)

    return pos_tag

def get_word_position(corpus):
    '''
    Given corpus, function is tagging every word regarding the position in the document - begin, midle or end.
    param input: corpus
    return: tagged words (tuple shape (word, position tag))

    '''
    tags = []
    for c in corpus:
        words = nltk.word_tokenize(c)
        word_number = len(words)
        list_begin = zip(words[:word_number//3], ['begin' for i in range(word_number//3)])
        l = list(list_begin)
        list_midle = zip(words[word_number//3:2*word_number//3], ['midle' for i in range(2*word_number//3 - word_number//3)])
        l.extend(list(list_midle))
        list_end = zip(words[2*word_number//3:], ['end' for i in range(word_number - 2*word_number//3)])
        l.extend(list(list_end))
        tags.append(l)

    return tags

def word_position_tagging(corpus):
    '''
    Given corpus, function extracts all tagged words, creates a vocabulary and appropriate model for ML algorithms.
    param input: corpus
    return: appropriate model for ML algorithms

    '''
    tags = get_word_position(corpus)
    vocabulary = create_vocabulary(tags)
    word_position_tag = create_model(tags, vocabulary)

    return word_position_tag

def compute_tf(corpus):
    '''
    Given corpus, function calculates frequency of every word in all documents in the corpus and returns a matrix of tf.
    param input: corpus
    return: matrix of term frequencies

    '''
    # Create vocabulary of the entire corpus
    vectorizer = CountVectorizer()
    c = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names()

    model = np.zeros((len(corpus), len(vocabulary)))
    for i, c in enumerate(corpus):
        # Extract words
        words = nltk.word_tokenize(c)
        document = np.zeros(len(vocabulary))
        # Create vocabulary of the specific document
        doc = vectorizer.fit_transform([c])
        doc_vocabulary = vectorizer.get_feature_names()
        doc_vocabulary_length = len(doc_vocabulary)
        for w in words:
            index = vocabulary.index(w) if w in vocabulary else -1
            if index != -1:
                document[index] += 1 / doc_vocabulary_length
        model[i] = document

    return model

def compute_tf_idf(corpus):
    '''
    Given corpus, function calculates the weight of rare words across all documents in the corpus and returns a matrix of tf-idf.
    param input: corpus
    return: matrix of term frequencies - inverse data frequencies

    '''
    vectorizer = TfidfVectorizer()
    c = vectorizer.fit_transform(corpus)

    return c.toarray()

def text_preprocessing(corpus, language):
    '''
    Given corpus and language indicator, function first removes punctuation and stopwords,
    stemms words and then creates different types of corpus representations:
    - bag of words model
    - unigram model
    - bigram model
    - bigram + unigram model
    - part of speech model
    - word position model
    - term frequency model
    - term frequency - inverse data frequency model
    param input: corpus, language indicator
    return: bag of words model, unigram model, bigram model, bigram + unigram model, part of speech model,
            word position model, term frequency model, term frequency - inverse data frequency model

    '''
    # Remove punctuation
    no_punctuation_corpus = remove_punctuation(corpus, language)

    # Remove stopwords
    no_stopwords_corpus = remove_stopwords(no_punctuation_corpus, language)

    # Stemming
    cleaned_corpus = stemming(no_stopwords_corpus, language)

    # Get the bag of words model
    bag_of_words_model = generate_ngrams(cleaned_corpus, (1, 1))
    logging.debug('Bag of words model for {} reviews:\n {}'.format(language, bag_of_words_model))

    # Get the unigram model
    unigram_model = generate_ngrams(no_punctuation_corpus, (1, 1))
    logging.debug('Unigram model for {} reviews:\n {}'.format(language, unigram_model))

    # Get the bigram model
    bigram_model = generate_ngrams(no_punctuation_corpus, (2, 2))
    logging.debug('Bigram model for {} reviews:\n {}'.format(language, bigram_model))

    # Get the bigram + unigram model
    bigram_unigram_model = generate_ngrams(no_punctuation_corpus, (1, 2))
    logging.debug('Bigram + unigram model for {} reviews:\n {}'.format(language, bigram_unigram_model))

    # Get the word position model
    word_position_model = word_position_tagging(cleaned_corpus)
    logging.debug('Word position model for {} reviews:\n {}'.format(language, word_position_model))

    # Get the term frequency model from bag of words model
    tf_model = compute_tf(cleaned_corpus)
    logging.debug('Term frequency model for {} reviews:\n {}'.format(language, tf_model))

    # Get the term frequency - inverse data frequency model
    tf_idf_model = compute_tf_idf(cleaned_corpus)
    logging.debug('Term frequency - inverse data frequency model for {} reviews:\n {}'.format(language, tf_idf_model))

    # Get the part of speech tag
    # TODO: change when POS for Serbian corpus is delivered
    if language == 'English':
        pos_tag_model = part_of_speech_tagging(cleaned_corpus)
        logging.debug('POS tag model for {} reviews:\n {}'.format(language, pos_tag_model))
        return bag_of_words_model, unigram_model, bigram_model, bigram_unigram_model, pos_tag_model, word_position_model, tf_model, tf_idf_model

    return bag_of_words_model, unigram_model, bigram_model, bigram_unigram_model, word_position_model, tf_model, tf_idf_model

def scaling(X_train, X_test):
    '''
    Given training and test dataset, function scales datasets in range [0, 1].
    param input: training and test dataset
    return: scaled training and test dataset

    '''
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test

def cross_validation(model, data, classes):
    '''
    Given model and dataset (data w/ matching class), function is doing 10-cross validation
    in order to predict how well the model will be trained on random dataset division.
    param input: model and dataset
    return: prediction of how well the model will be trained

    '''
    scores = model_selection.cross_val_score(model, data, classes, cv=10)
    logging.debug('Cross-validated scores: {}\n'.format(scores))

    return scores

def get_best_svm_hyperparameters(model_id):
    '''
    Given model id, function returns the best grid parameters for SVM classification.
    param input: model id
    return: best SVM hyperparameters for given model

    '''
    switcher = {
        'bow_srb': {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'},
        'unigram_srb': {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'},
        'bigram_srb': {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'},
        'bigram_unigram_srb': {'C': 0.1, 'gamma': 1, 'kernel': 'linear'},
        'position_srb': {'C': 10, 'gamma': 0.001, 'kernel': 'rbf'},
        'tf_srb': {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
        'tf_idf_srb': {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
        'bow_eng': {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'},
        'unigram_eng': {'C': 0.1, 'gamma': 1, 'kernel': 'linear'},
        'bigram_eng': {'C': 0.1, 'gamma': 1, 'kernel': 'linear'},
        'bigram_unigram_eng': {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'},
        'pos_tag_eng': {'C': 0.1, 'gamma': 1, 'kernel': 'linear'},
        'position_eng': {'C': 1000, 'gamma': 0.0001, 'kernel': 'rbf'},
        'tf_eng': {'C': 100, 'gamma': 0.0001, 'kernel': 'rbf'},
        'tf_idf_eng': {'C': 1, 'gamma': 1, 'kernel': 'linear'}
    }

    return switcher.get(model_id, "[SVM] Invalid model id\n")

def svm_classifier(data, classes, X_train, X_test, y_train, y_test, model_id):
    '''
    Given training and testing dataset and model id, functions creates a Support Vector Machine classifier with best hyperparameters,
    calculates the average of expected modeling accuracy using 10-cross validation (if '-c' flag is enabled),
    trains the model using the training set, predicts the response for the test dataset and returns the score between predicted and test dataset.
    If '-g' option is enabled, program will first do the grid search analysis with 3-cross validation
    and automatically fit the model with best hyperparameters.
    param input: dataset for training and testing, model id
    return: score between predicted and test dataset

    '''
    if args['grid_search'] == False:
        hyperparam_list = get_best_svm_hyperparameters(model_id)
        clf = svm.SVC(C=hyperparam_list['C'], gamma=hyperparam_list['gamma'], kernel=hyperparam_list['kernel'])
        if args['cross_validation'] == True:
            score = cross_validation(clf, data, classes)
            logging.info('[SVM] Expected accuracy: {}\n'.format(mean(score)))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    else:
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['rbf', 'linear']}

        grid = model_selection.GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)

        logging.debug('[SVM] Best grid parameters: {}'.format(grid.best_params_))
        logging.debug('[SVM] Best grid estimator: {}'.format(grid.best_estimator_))

        y_pred = grid.predict(X_test)

    return metrics.classification_report(y_test, y_pred)

def get_best_nb_hyperparameters(model_id):
    '''
    Given model id, function returns the best grid parameters for SVM classification.
    param input: model id
    return: best NB hyperparameters for given model

    '''
    switcher = {
        'bow_srb': {'alpha': 1.5, 'fit_prior': True},
        'unigram_srb': {'alpha': 1.5, 'fit_prior': True},
        'bigram_srb': {'alpha': 1.5, 'fit_prior': True},
        'bigram_unigram_srb': {'alpha': 1.5, 'fit_prior': True},
        'position_srb': {'alpha': 1.5, 'fit_prior': True},
        'tf_srb': {'alpha': 1.5, 'fit_prior': True},
        'tf_idf_srb': {'alpha': 1.5, 'fit_prior': True},
        'bow_eng': {'alpha': 1.5, 'fit_prior': True},
        'unigram_eng': {'alpha': 1.5, 'fit_prior': True},
        'bigram_eng': {'alpha': 1.5, 'fit_prior': True},
        'bigram_unigram_eng': {'alpha': 1.5, 'fit_prior': True},
        'pos_tag_eng': {'alpha': 1.0, 'fit_prior': True},
        'position_eng': {'alpha': 1.0, 'fit_prior': True},
        'tf_eng': {'alpha': 1.5, 'fit_prior': True},
        'tf_idf_eng': {'alpha': 1.5, 'fit_prior': True}
    }

    return switcher.get(model_id, "[NB] Invalid model id\n")

def nb_classifier(data, classes, X_train, X_test, y_train, y_test, model_id):
    '''
    Given training and testing dataset and model id, functions creates a Naive Bayes classifier with best hyperparameters,
    calculates the average of expected modeling accuracy using 10-cross validation (if '-c' flag is enabled),
    trains the model using the training set, predicts the response for the test dataset and returns the score between predicted and test dataset.
    If '-g' option is enabled, program will first do the grid search analysis with 3-cross validation
    and automatically fit the model with best hyperparameters.
    param input: dataset for training and testing, model id
    return: score between predicted and test dataset

    '''
    if args['grid_search'] == False:
        hyperparam_list = get_best_nb_hyperparameters(model_id)
        clf = naive_bayes.MultinomialNB(alpha=hyperparam_list['alpha'], fit_prior=hyperparam_list['fit_prior'])
        if args['cross_validation'] == True:
            score = cross_validation(clf, data, classes)
            logging.info('[NB] Expected accuracy: {}\n'.format(mean(score)))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    else:
        param_grid = {'alpha': [0.5, 1.0, 1.5],
                      'fit_prior': [True, False]}

        grid = model_selection.GridSearchCV(naive_bayes.MultinomialNB(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)

        logging.debug('[NB] Best grid parameters: {}'.format(grid.best_params_))
        logging.debug('[NB] Best grid estimator: {}'.format(grid.best_estimator_))

        y_pred = grid.predict(X_test)

    return metrics.classification_report(y_test, y_pred)

def get_best_mlp_hyperparameters(model_id):
    '''
    Given model id, function returns the best grid parameters for MLP classification.
    param input: model id
    return: best MLP hyperparameters for given model

    '''
    switcher = {
        'bow_srb': {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'sgd', 'alpha': 0.05},
        'unigram_srb': {'hidden_layer_sizes': (50,50,50), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.05},
        'bigram_srb': {'hidden_layer_sizes': (50,50,50), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.05},
        'bigram_unigram_srb': {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.05},
        'position_srb': {'hidden_layer_sizes': (50,50,50), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.0001},
        'tf_srb': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'sgd', 'alpha': 0.05},
        'tf_idf_srb': {'hidden_layer_sizes': (50,100,50), 'activation': 'tanh', 'solver': 'lbfgs', 'alpha': 0.05},
        'bow_eng': {'hidden_layer_sizes': (100,), 'activation': 'tanh', 'solver': 'lbfgs', 'alpha': 0.0001},
        'unigram_eng': {'hidden_layer_sizes': (50,50,50), 'activation': 'tanh', 'solver': 'lbfgs', 'alpha': 0.0001},
        'bigram_eng': {'hidden_layer_sizes': (50,50,50), 'activation': 'tanh', 'solver': 'lbfgs', 'alpha': 0.0001},
        'bigram_unigram_eng': {'hidden_layer_sizes': (50,100,50), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.05},
        'pos_tag_eng': {'hidden_layer_sizes': (50,100,50), 'activation': 'tanh', 'solver': 'adam', 'alpha': 0.05},
        'position_eng': {'hidden_layer_sizes': (50,50,50), 'activation': 'relu', 'solver': 'adam', 'alpha': 0.05},
        'tf_eng': {'hidden_layer_sizes': (50,50,50), 'activation': 'tanh', 'solver': 'sgd', 'alpha': 0.05},
        'tf_idf_eng': {'hidden_layer_sizes': (50,100,50), 'activation': 'tanh', 'solver': 'sgd', 'alpha': 0.0001},
    }

    return switcher.get(model_id, "[MLP] Invalid model id\n")

def mlp_classifier(data, classes, X_train, X_test, y_train, y_test, model_id):
    '''
    Given training and testing dataset and model id, functions creates a Multi Layer Perceptron classifier with best hyperparameters,
    calculates the average of expected modeling accuracy using 10-cross validation (if '-c' flag is enabled),
    trains the model using the training set, predicts the response for the test dataset and returns the score between predicted and test dataset.
    If '-g' option is enabled, program will first do the grid search analysis with 3-cross validation
    and automatically fit the model with best hyperparameters.
    param input: dataset for training and testing, model id
    return: score between predicted and test dataset

    '''
    if args['grid_search'] == False:
        hyperparam_list = get_best_mlp_hyperparameters(model_id)
        clf = neural_network.MLPClassifier(hidden_layer_sizes=hyperparam_list['hidden_layer_sizes'],
            activation=hyperparam_list['activation'], solver=hyperparam_list['solver'], alpha=hyperparam_list['alpha'])
        if args['cross_validation'] == True:
            score = cross_validation(clf, data, classes)
            logging.info('[MLP] Expected accuracy: {}\n'.format(mean(score)))

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    else:
        param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                      'activation': ['tanh', 'relu'],
                      'solver': ['sgd', 'adam', 'lbfgs'],
                      'alpha': [0.0001, 0.05]}

        grid = model_selection.GridSearchCV(neural_network.MLPClassifier(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)

        logging.debug('[MLP] Best grid parameters: {}'.format(grid.best_params_))
        logging.debug('[MLP] Best grid estimator: {}'.format(grid.best_estimator_))

        y_pred = grid.predict(X_test)

    return metrics.classification_report(y_test, y_pred)

def classification(data, classes, model_id):
    '''
    Given data, classes and size parameter, function splits the data in groups for training and testing (size for test group is test_size).
    Afterwards, function scales the training and test dataset, trains the model data using SVM, NB and MLP algorithms
    and calculates the accuracy between predicted and test data.
    param input: data, classes and size parameter
    return: None

    '''
    test_size = int(args['test_percentage']) * 0.01 if args['test_percentage'] != None else 0.2

    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, classes, test_size=test_size)

    X_train, X_test = scaling(X_train, X_test)

    score = svm_classifier(data, classes, X_train, X_test, y_train, y_test, model_id)
    logging.info('[SVM] Accuracy: {}\n'.format(score))

    score = nb_classifier(data, classes, X_train, X_test, y_train, y_test, model_id)
    logging.info('[NB] Accuracy: {}\n'.format(score))

    score = mlp_classifier(data, classes, X_train, X_test, y_train, y_test, model_id)
    logging.info('[MLP] Accuracy: {}\n'.format(score))

    gc.collect()

if __name__ == '__main__':

    # Set the argument parser
    args = get_parser()

    # Set logging level
    set_logging_level(args)

    # Install nltk dependencies
    nltk_dependencies()

    # Get the datasets for Serbian and English reviews
    corpus_srb, classes_srb = get_srb_corpus()
    corpus_eng, classes_eng = get_eng_corpus()

    # Get different data representations: bag of words, unigram model, bigram model, bigram + unigram model,
    # part of speech tagging, word position tagging, tf model, tf-idf model
    bow_srb, unigram_srb, bigram_srb, bigram_unigram_srb, position_srb, tf_srb, tf_idf_srb = text_preprocessing(corpus_srb, 'Serbian')
    bow_eng, unigram_eng, bigram_eng, bigram_unigram_eng, pos_tag_eng, position_eng, tf_eng, tf_idf_eng = text_preprocessing(corpus_eng, 'English')

    logging.info(' --- Serbian reviews (Bag Of Words Model) --- \n')
    classification(bow_srb, classes_srb, "bow_srb")
    logging.info(' --- Serbian reviews (Unigram Model) --- \n')
    classification(unigram_srb, classes_srb, "unigram_srb")
    logging.info(' --- Serbian reviews (Bigram Model) --- \n')
    classification(bigram_srb, classes_srb, "bigram_srb")
    logging.info(' --- Serbian reviews (Bigram + Unigram Model) --- \n')
    classification(bigram_unigram_srb, classes_srb, "bigram_unigram_srb")
    logging.info(' --- Serbian reviews (Word Position Model) --- \n')
    classification(position_srb, classes_srb, "position_srb")
    logging.info(' --- Serbian reviews (Term Frequency Model) --- \n')
    classification(tf_srb, classes_srb, "tf_srb")
    logging.info(' --- Serbian reviews (Term Frequency - Inverse Data Frequency Model) --- \n')
    classification(tf_idf_srb, classes_srb, "tf_idf_srb")

    logging.info(' --- English reviews (Bag Of Words Model) --- \n')
    classification(bow_eng, classes_eng, "bow_eng")
    logging.info(' --- English reviews (Unigram Model) --- \n')
    classification(unigram_eng, classes_eng, "unigram_eng")
    logging.info(' --- English reviews (Bigram Model) --- \n')
    classification(bigram_eng, classes_eng, "bigram_eng")
    logging.info(' --- English reviews (Bigram + Unigram Model) --- \n')
    classification(bigram_unigram_eng, classes_eng, "bigram_unigram_eng")
    logging.info(' --- English reviews (Part Of Speech Model) --- \n')
    classification(pos_tag_eng, classes_eng, "pos_tag_eng")
    logging.info(' --- English reviews (Word Position Model) --- \n')
    classification(position_eng, classes_eng, "position_eng")
    logging.info(' --- English reviews (Term Frequency Model) --- \n')
    classification(tf_eng, classes_eng, "tf_eng")
    logging.info(' --- English reviews (Term Frequency - Inverse Data Frequency Model) --- \n')
    classification(tf_idf_eng, classes_eng, "tf_idf_eng")
