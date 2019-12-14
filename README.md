# Sentiment-Text-Analysis-using-Machine-Learning-Techniques

## Project structure
- src/sentiment_analysis.py is the main file
- src/helper folder contains all helper functions (stemmer fo Serbian language)
- src/test folder contains unit tests for funcionalities in sentiment_analysis.py file
- src/data folder should contain input data

### Corpus representations
- There are two movie review corpuses
  - Serbian corpus: https://github.com/vukbatanovic/SerbMR (SerbMR-3C - csv format)
    - positive (841), negative (841) and neutral (841) reviews
  - English corpus: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
    - positive (1000) and negative (1000) reviews

- Corpuses are represented in different styles:
  - Bag of words (BOW) model
  - Bigram model
  - Bigram + unigram model
  - Part of speech (POS) model
  - Word position in text model
  - Term frequency model (for unigrams)
  - Term frequency - inverse data frequency model (for unigrams)

- Additional:
  - Corpuses are cleaned from punctuation
  - Words are reduced to their root form (stemming)

### Machine learning techniques
- All models are trained and tested using two algorithms:
  - SVM (Support Vector Machine) algorithm
  - NB (Naive Bayes) algorithm
- Before training, 10-cross validation is implemented to estimate the fitting model and ML algorithm

## Dependencies:
- Python 3.7.4 (64 bit) or above
- Libraries:
  - os
  - logging
  - argparse
  - pandas
  - nltk
  - numpy
  - string
  - sklearn
  - statistics

## Usage:
- Download the corpuses (unzip if necessary) and locate them in src/data folder
- Run the code by typing: "python sentiment_analysis.py"
  - set the logging level e.g. "python sentiment_analysis.py -l debug" ([critical, error, warning, info, debug])
  - set the percentage number for test data e.g. "python sentiment_analysis.py -t 30" ([0-100])
  - set the option for 10-cross validation e.g. "python sentiment_analysis.py -c"
  - type "python sentiment_analysis.py --help" to show all argument options
- Testing: go to src/test folder, type "python test_sentiment_analysis.py"
