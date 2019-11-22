# coding=UTF-8

import unittest
import numpy as np

import sys
sys.path.insert(0, '../')
import sentiment_analysis as sa

corpus_srb = ["Film 'Kum' mi se uopšte ne dopada. Užasno je dosadan, dug i nezanimljiv!",
              "Preporučujem svima da pogledaju film 'Memento'. Oduševljen sam njime."]
corpus_eng = ["I don't like movie 'The Godfather' at all. It is terribly boring, long and uninteresting!",
              "I recommend everyone to watch the movie 'Memento'. I'm thrilled with it."]

class TestPreprocessFunctionalities(unittest.TestCase):

    def test_remove_punctuation(self):
        cleaned_corpus_srb = sa.remove_punctuation(corpus_srb)
        cleaned_corpus_eng = sa.remove_punctuation(corpus_eng)

        expected_srb = ["Film Kum mi se uopšte ne dopada Užasno je dosadan dug i nezanimljiv",
                        "Preporučujem svima da pogledaju film Memento Oduševljen sam njime"]
        expected_eng = ["I dont like movie The Godfather at all It is terribly boring long and uninteresting",
                        "I recommend everyone to watch the movie Memento Im thrilled with it"]

        self.assertTrue((cleaned_corpus_srb == expected_srb) and (cleaned_corpus_eng == expected_eng))

    def test_stemming(self):
        # TODO: add for Serbian movies when stemmer works fine
        stemmed_corpus_eng = sa.stemming(corpus_eng, 'English')

        expected_eng = ["I do n't like movi 'the godfath ' at all . It is terribl bore , long and uninterest ! ",
                        "I recommend everyon to watch the movi 'memento ' . I 'm thrill with it . "]

        self.assertTrue(stemmed_corpus_eng == expected_eng)

    def test_generate_ngrams_unigrams(self):
        unigrams_srb = sa.generate_ngrams(corpus_srb, 1)
        unigrams_eng = sa.generate_ngrams(corpus_eng, 1)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                                 [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0]])
        expected_eng = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]])

        self.assertTrue((np.array_equal(unigrams_srb, expected_srb)) and (np.array_equal(unigrams_eng, expected_eng)))

    def test_generate_ngrams_bigrams(self):
        bigrams_srb = sa.generate_ngrams(corpus_srb, 2)
        bigrams_eng = sa.generate_ngrams(corpus_eng, 2)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
                                 [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]])
        expected_eng = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1]])

        self.assertTrue((np.array_equal(bigrams_srb, expected_srb)) and (np.array_equal(bigrams_eng, expected_eng)))

    def test_get_part_of_speech_words(self):
        # TODO: add for Serbian movies when POS tagger is delivered
        pos_tag_eng = sa.get_part_of_speech_words(corpus_eng)

        expected_eng = [[('I', 'PRP'), ('do', 'VBP'), ("n't", 'RB'), ('like', 'VB'), ('movie', 'NN'), ("'The", 'IN'), ('Godfather', 'NNP'), ("'", 'POS'), ('at', 'IN'), ('all', 'DT'),
                         ('.', '.'), ('It', 'PRP'), ('is', 'VBZ'), ('terribly', 'RB'), ('boring', 'JJ'), (',', ','), ('long', 'JJ'), ('and', 'CC'), ('uninteresting', 'JJ'), ('!', '.')],
                        [('I', 'PRP'), ('recommend', 'VBP'), ('everyone', 'NN'), ('to', 'TO'), ('watch', 'VB'), ('the', 'DT'), ('movie', 'NN'), ("'Memento", 'POS'), ("'", 'POS'), ('.', '.'),
                         ('I', 'PRP'), ("'m", 'VBP'), ('thrilled', 'JJ'), ('with', 'IN'), ('it', 'PRP'), ('.', '.')]]

        self.assertTrue(pos_tag_eng == expected_eng)

    def test_part_of_speech_tagging(self):
        # TODO: add for Serbian movies when POS tagger is delivered
        pos_tag_eng = sa.part_of_speech_tagging(corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_eng = np.array([[1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                [0., 1., 1., 0., 1., 0., 2., 0., 2., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])

        self.assertTrue(np.allclose(pos_tag_eng, expected_eng))

    def test_get_word_position(self):
        word_position_srb = sa.get_word_position(corpus_srb)
        word_position_eng = sa.get_word_position(corpus_eng)

        expected_srb = [[('Film', 'begin'), ("'Kum", 'begin'), ("'", 'begin'), ('mi', 'begin'), ('se', 'begin'),
                         ('uopšte', 'midle'), ('ne', 'midle'), ('dopada', 'midle'), ('.', 'midle'), ('Užasno', 'midle'), ('je', 'midle'),
                        ('dosadan', 'end'), (',', 'end'), ('dug', 'end'), ('i', 'end'), ('nezanimljiv', 'end'), ('!', 'end')],
                        [('Preporučujem', 'begin'), ('svima', 'begin'), ('da', 'begin'), ('pogledaju', 'begin'),
                        ('film', 'midle'), ("'Memento", 'midle'), ("'", 'midle'), ('.', 'midle'),
                        ('Oduševljen', 'end'), ('sam', 'end'), ('njime', 'end'), ('.', 'end')]]
        expected_eng = [[('I', 'begin'), ('do', 'begin'), ("n't", 'begin'), ('like', 'begin'), ('movie', 'begin'), ("'The", 'begin'),
                         ('Godfather', 'midle'), ("'", 'midle'), ('at', 'midle'), ('all', 'midle'), ('.', 'midle'), ('It', 'midle'), ('is', 'midle'),
                         ('terribly', 'end'), ('boring', 'end'), (',', 'end'), ('long', 'end'), ('and', 'end'), ('uninteresting', 'end'), ('!', 'end')],
                        [('I', 'begin'), ('recommend', 'begin'), ('everyone', 'begin'), ('to', 'begin'), ('watch', 'begin'),
                         ('the', 'midle'), ('movie', 'midle'), ("'Memento", 'midle'), ("'", 'midle'), ('.', 'midle'),
                         ('I', 'end'), ("'m", 'end'), ('thrilled', 'end'), ('with', 'end'), ('it', 'end'), ('.', 'end')]]

        self.assertTrue((word_position_srb == expected_srb) and (word_position_eng == expected_eng))

    def test_word_position_tagging(self):
        position_srb = sa.word_position_tagging(corpus_srb)
        position_eng = sa.word_position_tagging(corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1.],
                                 [0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0.]])
        expected_eng = np.array([[1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                 [0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])

        self.assertTrue((np.allclose(position_srb, expected_srb)) and (np.allclose(position_eng, expected_eng)))

    def test_create_vocabulary(self):
        # TODO: add for Serbian movies when POS tagger is delivered
        tagged_position_corpus_srb = sa.get_word_position(corpus_srb)
        tagged_position_corpus_eng = sa.get_word_position(corpus_eng)
        tagged_pos_tag_corpus_eng = sa.get_part_of_speech_words(corpus_eng)

        vocabulary_position_srb = sa.create_vocabulary(tagged_position_corpus_srb)
        vocabulary_position_eng = sa.create_vocabulary(tagged_position_corpus_eng)
        vocabulary_pos_tag_eng = sa.create_vocabulary(tagged_pos_tag_corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_position_srb = [('!', 'end'), ("'", 'begin'), ("'", 'midle'), ("'Kum", 'begin'), ("'Memento", 'midle'), (',', 'end'), ('.', 'end'), ('.', 'midle'), ('Film', 'begin'), ('Oduševljen', 'end'), ('Preporučujem', 'begin'), ('Užasno', 'midle'), ('da', 'begin'), ('dopada', 'midle'),
                                 ('dosadan', 'end'), ('dug', 'end'), ('film', 'midle'), ('i', 'end'), ('je', 'midle'), ('mi', 'begin'), ('ne', 'midle'), ('nezanimljiv', 'end'), ('njime', 'end'), ('pogledaju', 'begin'), ('sam', 'end'), ('se', 'begin'), ('svima', 'begin'), ('uopšte', 'midle')]
        expected_position_eng = [('!', 'end'), ("'", 'midle'), ("'Memento", 'midle'), ("'The", 'begin'), ("'m", 'end'), (',', 'end'), ('.', 'end'), ('.', 'midle'), ('Godfather', 'midle'), ('I', 'begin'), ('I', 'end'), ('It', 'midle'), ('all', 'midle'), ('and', 'end'), ('at', 'midle'), ('boring', 'end'), ('do', 'begin'), ('everyone', 'begin'),
                                 ('is', 'midle'), ('it', 'end'), ('like', 'begin'), ('long', 'end'), ('movie', 'begin'), ('movie', 'midle'), ("n't", 'begin'), ('recommend', 'begin'), ('terribly', 'end'), ('the', 'midle'), ('thrilled', 'end'), ('to', 'begin'), ('uninteresting', 'end'), ('watch', 'begin'), ('with', 'end')]
        expected_pos_tag_eng = [('!', '.'), ("'", 'POS'), ("'Memento", 'POS'), ("'The", 'IN'), ("'m", 'VBP'), (',', ','), ('.', '.'), ('Godfather', 'NNP'), ('I', 'PRP'), ('It', 'PRP'), ('all', 'DT'), ('and', 'CC'), ('at', 'IN'), ('boring', 'JJ'), ('do', 'VBP'), ('everyone', 'NN'),
                                ('is', 'VBZ'), ('it', 'PRP'), ('like', 'VB'), ('long', 'JJ'), ('movie', 'NN'), ("n't", 'RB'), ('recommend', 'VBP'), ('terribly', 'RB'), ('the', 'DT'), ('thrilled', 'JJ'), ('to', 'TO'), ('uninteresting', 'JJ'), ('watch', 'VB'), ('with', 'IN')]

        self.assertTrue((vocabulary_position_srb == expected_position_srb) and (vocabulary_position_eng == expected_position_eng) and
                        (vocabulary_pos_tag_eng == expected_pos_tag_eng))

    def test_create_model(self):
        # TODO: add for Serbian movies when POS tagger is delivered
        tagged_position_corpus_srb = sa.get_word_position(corpus_srb)
        tagged_position_corpus_eng = sa.get_word_position(corpus_eng)
        tagged_pos_tag_corpus_eng = sa.get_part_of_speech_words(corpus_eng)

        vocabulary_position_srb = sa.create_vocabulary(tagged_position_corpus_srb)
        vocabulary_position_eng = sa.create_vocabulary(tagged_position_corpus_eng)
        vocabulary_pos_tag_eng = sa.create_vocabulary(tagged_pos_tag_corpus_eng)

        position_model_srb = sa.create_model(tagged_position_corpus_srb, vocabulary_position_srb)
        position_model_eng = sa.create_model(tagged_position_corpus_eng, vocabulary_position_eng)
        pos_tagging_model_eng = sa.create_model(tagged_pos_tag_corpus_eng, vocabulary_pos_tag_eng)

        # NOTE: arrays are in alphabetical order
        expected_position_srb = np.array([[1., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1.],
                                          [0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0.]])
        expected_position_eng = np.array([[1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                          [0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])
        expected_pos_tag_eng = np.array([[1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                         [0., 1., 1., 0., 1., 0., 2., 0., 2., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])

        self.assertTrue((np.allclose(position_model_srb, expected_position_srb)) and (np.allclose(position_model_eng, expected_position_eng)) and
                        (np.allclose(pos_tagging_model_eng, expected_pos_tag_eng)))

    def test_compute_tf(self):
        bow_srb = sa.generate_ngrams(corpus_srb, 1)
        bow_eng = sa.generate_ngrams(corpus_eng, 1)

        tf_srb = sa.compute_tf(bow_srb)
        tf_eng = sa.compute_tf(bow_eng)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0., 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0., 0.05, 0.05, 0.05, 0., 0., 0., 0., 0., 0.05, 0., 0.05, 0.05],
                                [0.05, 0., 0., 0., 0.05, 0., 0., 0.05, 0., 0., 0., 0.05, 0.05, 0.05, 0.05, 0.05, 0., 0.05, 0., 0.]])
        expected_eng = np.array([[0.04761905, 0.04761905, 0.04761905, 0.04761905, 0.04761905, 0., 0.04761905, 0.04761905, 0.04761905, 0.04761905, 0.04761905, 0., 0.04761905, 0., 0.04761905, 0.04761905, 0., 0., 0.04761905, 0., 0.],
                                [0., 0., 0., 0., 0., 0.04761905, 0., 0., 0.04761905, 0., 0., 0.04761905, 0.04761905, 0.04761905, 0., 0.04761905, 0.04761905, 0.04761905, 0., 0.04761905, 0.04761905]])

        self.assertTrue((np.allclose(tf_srb, expected_srb)) and (np.allclose(tf_eng, expected_eng)))

    def test_compute_tf_idf(self):
        tf_idf_srb = sa.compute_tf_idf(corpus_srb)
        tf_idf_eng = sa.compute_tf_idf(corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0., 0.29480389, 0.29480389, 0.29480389, 0.2097554, 0.29480389, 0.29480389, 0., 0.29480389, 0.29480389, 0.29480389, 0., 0., 0., 0., 0., 0.29480389, 0., 0.29480389, 0.29480389],
                                 [0.34287126, 0., 0., 0., 0.24395573, 0., 0., 0.34287126, 0., 0., 0., 0.34287126, 0.34287126, 0.34287126, 0.34287126, 0.34287126, 0., 0.34287126, 0., 0.]])
        expected_eng = np.array([[0.28263102, 0.28263102, 0.28263102, 0.28263102, 0.28263102, 0., 0.28263102, 0.28263102, 0.2010943, 0.28263102, 0.28263102, 0., 0.2010943, 0., 0.28263102, 0.2010943, 0., 0., 0.28263102, 0., 0.],
                                 [0., 0., 0., 0., 0., 0.34261985, 0., 0., 0.24377685, 0., 0., 0.34261985, 0.24377685, 0.34261985, 0., 0.24377685, 0.34261985, 0.34261985, 0., 0.34261985, 0.34261985]])

        self.assertTrue((np.allclose(tf_idf_srb, expected_srb)) and (np.allclose(tf_idf_eng, expected_eng)))

if __name__ == '__main__':
    unittest.main()
