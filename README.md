# Sentiment-Text-Analysis-using-Machine-Learning-Techniques

## File sentiment_analysis.py:
- For input has two movie review corpuses
  - Serbian corpus: https://github.com/vukbatanovic/SerbMR (SerbMR-3C - csv format) - positive, negative and neutral classes
  - English corpus: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz - positive and negative classes
- Creates a bag of words model for each corpus
- Bag of words models are trained and tested using two algorithms:
  - SVM (Support Vector Machine)
  - NB (Naive Bayes)
  
## Dependencies:
- Python 3.7.4 (64 bit) or above
- Libraries:
  - os
  - logging
  - argparse
  - pandas
  - sklearn
  
## Usage:
- Download the corpuses and locate them in folder where sentiment_analysis.py file is located (unzip if necessary)
- python sentiment_analysis.py
  - set the logging level and percentage for test data e.g. python sentiment_analysis.py -l debug -t 30
  - python sentiment_analysis.py --help (to show all argument options)