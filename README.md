# Sentiment-Text-Analysis-using-Machine-Learning-Techniques

## File sentiment_analysis.py
- For input has two movie review corpuses
  - Serbian corpus: https://github.com/vukbatanovic/SerbMR (SerbMR-3C - csv format) - positive, negative and neutral classes
  - English corpus: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz - positive and negative classes
- Corpuses are represented in different styles:
  - Bag of words (BOW) model
  - Bigram model
  - Part of speech (POS) model
  - Word position in text model
  - Term frequency model (for unigrams)
  - Term frequency - inverse data frequency model (for unigrams)
- All models are trained and tested using two algorithms:
  - SVM (Support Vector Machine)
  - NB (Naive Bayes)

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

## Usage:
- Download the corpuses and locate them in folder where sentiment_analysis.py file is located (unzip if necessary)
- Run the code by typing: "python sentiment_analysis.py"
  - set the logging level e.g. "python sentiment_analysis.py -l debug" ([CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET])
  - set the percentage number for test data e.g. "python sentiment_analysis.py -t 30" ([0-100])
  - type "python sentiment_analysis.py --help" to show all argument options