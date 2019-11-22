import os
import logging
import argparse
import nltk
import string
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import svm
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from helper import serbian_stemmer as ss

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
    arguments = vars(parser.parse_args())

    return arguments

def set_loging_level(args):
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

    return data.Text, data.Rating

def remove_punctuation(corpus):
    '''
    Given corpus, function removes punctuation in all corpus documents.
    param input: corpus
    return: corpus w/o punctuation

    '''
    cleaned_text = []
    replacer = str.maketrans(dict.fromkeys(string.punctuation))
    for c in corpus:
        cleaned_text.append(c.translate(replacer))

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
        # TODO: find the word first in the vocabulary when it is delivered
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
    Given corpus and number n (n stands for ngram), function extracts all ngrams from all files in corpus.
    For n = 1 this function creates a bag of words model.
    For n = 2 this function creates a bigram model.
    param input: corpus, number n
    return: vector of ngrams

    '''
    vectorizer = CountVectorizer(ngram_range=(n,n), max_features=100000)
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
    Given bow corpus model, function calculates frequency of every word in all documents in the corpus and returns a matrix of tf.
    param input: corpus
    return: matrix of term frequencies

    '''
    tf_matrix = np.zeros(corpus.shape)
    for i in range(len(corpus)):
        for j in range(len(corpus[i])):
            tf_matrix[i][j] = corpus[i][j] / len(corpus[i])

    return tf_matrix

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
    Given corpus and language indicator, function first removes punctuation, stemms words and then creates different types of corpus representations:
    - bag of words model
    - bigram model
    - part of speech model
    - word position model
    - term frequency model
    - term frequency - inverse data frequency model
    param input: corpus, language indicator
    return: bag of words model, bigram model, part of speech model, word position model,
            term frequency model, term frequency - inverse data frequency model

    '''
    # Remove punctuation
    cleaned_corpus = remove_punctuation(corpus)

    # Stemming
    stemmed_corpus = stemming(cleaned_corpus, language)

    # Get the bag of words model
    bag_of_words_model = generate_ngrams(stemmed_corpus, 1)
    logging.debug('Bag of words for {} reviews:\n {}'.format(language, bag_of_words_model))

    # Get the bigram model
    bigram_model = generate_ngrams(stemmed_corpus, 2)
    logging.debug('Bigram model for {} reviews:\n {}'.format(language, bigram_model))

    # Get the word position model
    word_position_model = word_position_tagging(stemmed_corpus)
    logging.debug('Word position model for {} reviews:\n {}'.format(language, word_position_model))

    # Get the term frequency model from bag of words model
    tf_model = compute_tf(bag_of_words_model)
    logging.debug('Term frequency model for {} reviews:\n {}'.format(language, tf_model))

    # Get the term frequency - inverse data frequency model
    tf_idf_model = compute_tf_idf(stemmed_corpus)
    logging.debug('Term frequency - inverse data frequency model for {} reviews:\n {}'.format(language, tf_idf_model))

    # Get the part of speech tag
    # TODO: change when POS for Serbian corpus is delivered
    if language == 'English':
        pos_tag_model = part_of_speech_tagging(stemmed_corpus)
        logging.debug('POS tag model for {} reviews:\n {}'.format(language, pos_tag_model))
        return bag_of_words_model, bigram_model, pos_tag_model, word_position_model, tf_model, tf_idf_model

    return bag_of_words_model, bigram_model, word_position_model, tf_model, tf_idf_model

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

def classification(data, classes, test_size):
    '''
    Given data, classes and size parameter, function splits the data in groups for training and testing (size for test group is test_size).
    Afterwards, function trains the model data using SVM and NB algorithms and calculates the accuracy between predicted and test data.
    param input: data, classes and size parameter
    return: None

    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, classes, test_size=test_size)

    score = svm_classifier(X_train, X_test, y_train, y_test)
    logging.info('SVM accuracy: {}\n'.format(score))

    score = nb_classifier(X_train, X_test, y_train, y_test)
    logging.info('NB accuracy: {}\n'.format(score))

if __name__ == '__main__':

    # Set the argument parser
    args = get_parser()

    # Set logging level
    set_loging_level(args)

    # Install nltk dependencies
    nltk_dependencies()

    # Get the datasets for Serbian and English reviews
    corpus_srb, classes_srb = get_srb_corpus()
    corpus_eng, classes_eng = get_eng_corpus()

    # Get different data representations: bag of words, bigrams, part of speech tagging, word position tagging, tf model, tf-idf model
    bow_srb, bigram_srb, position_srb, tf_srb, tf_idf_srb = text_preprocessing(corpus_srb, 'Serbian')
    bow_eng, bigram_eng, pos_tag_eng, position_eng, tf_eng, tf_idf_eng = text_preprocessing(corpus_eng, 'English')

    # Classification
    test_size = int(args['test_percentage']) * 0.01 if args['test_percentage'] != None else 0.3

    logging.info(' --- Serbian reviews (Bag Of Words Model) --- \n')
    classification(bow_srb, classes_srb, test_size)
    logging.info(' --- Serbian reviews (Bigram Model) --- \n')
    classification(bigram_srb, classes_srb, test_size)
    logging.info(' --- Serbian reviews (Word Position Model) --- \n')
    classification(position_srb, classes_srb, test_size)
    logging.info(' --- Serbian reviews (Term Frequency Model) --- \n')
    classification(tf_srb, classes_srb, test_size)
    logging.info(' --- Serbian reviews (Term Frequency - Inverse Data Frequency Model) --- \n')
    classification(tf_idf_srb, classes_srb, test_size)

    logging.info(' --- English reviews (Bag Of Words Model) --- \n')
    classification(bow_eng, classes_eng, test_size)
    logging.info(' --- English reviews (Bigram Model) --- \n')
    classification(bigram_eng, classes_eng, test_size)
    logging.info(' --- English reviews (Part Of Speech Model) --- \n')
    classification(pos_tag_eng, classes_eng, test_size)
    logging.info(' --- English reviews (Word Position Model) --- \n')
    classification(position_eng, classes_eng, test_size)
    logging.info(' --- English reviews (Term Frequency Model) --- \n')
    classification(tf_eng, classes_eng, test_size)
    logging.info(' --- English reviews (Term Frequency - Inverse Data Frequency Model) --- \n')
    classification(tf_idf_eng, classes_eng, test_size)
