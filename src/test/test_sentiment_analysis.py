# coding=UTF-8

import unittest
import numpy as np

import sys
sys.path.insert(0, '../')
import sentiment_analysis as sa

corpus_srb = ["Film predstavlja odu životnom stilu jednog pacifiste.",
              "The 'Big Lewbowski' je klasična priča prevare, kriminala i spletkarenja viđena kroz oči skromnog čoveka."]
corpus_eng = ["I don't like movie 'The Godfather' at all. It is terribly boring, long and uninteresting!",
              "I recommend everyone to watch the movie 'Memento'. I'm thrilled with it."]
corpus_srb_cyrilic = ["Филм представља оду животном стилу једног пацифисте.",
                      "Тхе 'Big Lewbowski' је класична прича преваре, криминала и сплеткарења виђена кроз очи скромног човека."]

class TestPreprocessFunctionalities(unittest.TestCase):

    def test_convert_to_latin(self):
        converted_corpus_srb = sa.convert_to_latin(corpus_srb_cyrilic)

        expected_srb = ["film predstavlja odu zxivotnom stilu jednog pacifiste.",
                        "the 'big lewbowski' je klasicxna pricxa prevare, kriminala i spletkarenja vidxena kroz ocxi skromnog cxoveka."]

        self.assertTrue(converted_corpus_srb == expected_srb)

    def test_remove_punctuation(self):
        cleaned_corpus_srb = sa.remove_punctuation(corpus_srb, 'Serbian')
        cleaned_corpus_eng = sa.remove_punctuation(corpus_eng, 'English')

        expected_srb = ["Film predstavlja odu životnom stilu jednog pacifiste",
                        "The Big Lewbowski je klasična priča prevare kriminala i spletkarenja viđena kroz oči skromnog čoveka"]
        expected_eng = ["I dont like movie The Godfather at all It is terribly boring long and uninteresting",
                        "I recommend everyone to watch the movie Memento Im thrilled with it"]

        self.assertTrue((cleaned_corpus_srb == expected_srb) and (cleaned_corpus_eng == expected_eng))

    def test_remove_stopwords(self):
        cleaned_corpus_srb = sa.remove_stopwords(corpus_srb, 'Serbian')
        cleaned_corpus_eng = sa.remove_stopwords(corpus_eng, 'English')

        expected_srb = ["Film predstavlja odu životnom stilu pacifiste.",
                        "The 'Big Lewbowski' klasična priča prevare, kriminala spletkarenja viđena oči skromnog čoveka."]
        expected_eng = ["I like movie 'The Godfather' all. It terribly boring, long uninteresting!",
                        "I recommend everyone watch movie 'Memento'. I'm thrilled it."]

        self.assertTrue((cleaned_corpus_srb == expected_srb) and (cleaned_corpus_eng == expected_eng))

    def test_word_normalization(self):
        stemmed_corpus_srb = sa.convert_to_latin(corpus_srb)
        stemmed_corpus_srb = sa.remove_punctuation(stemmed_corpus_srb, 'Serbian')
        stemmed_corpus_srb = sa.word_normalization(stemmed_corpus_srb, 'Serbian')
        stemmed_corpus_eng = sa.word_normalization(corpus_eng, 'English')

        expected_srb = ["film predstavlja oda zxivotan stilo jedan pacifista ",
                        "the big lewbowski ona klasicxan pricxa prevara kriminal i spletkarenje viditi kroz ocxi skroman cxovek "]
        expected_eng = ["I do n't like movi 'the godfath ' at all . It is terribl bore , long and uninterest ! ",
                        "I recommend everyon to watch the movi 'memento ' . I 'm thrill with it . "]

        self.assertTrue((stemmed_corpus_srb == expected_srb) and (stemmed_corpus_eng == expected_eng))

    def test_generate_ngrams_unigrams(self):
        unigrams_srb = sa.generate_ngrams(corpus_srb, (1, 1))
        unigrams_eng = sa.generate_ngrams(corpus_eng, (1, 1))

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                 [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0]])
        expected_eng = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]])

        self.assertTrue((np.array_equal(unigrams_srb, expected_srb)) and (np.array_equal(unigrams_eng, expected_eng)))

    def test_generate_ngrams_bigrams(self):
        bigrams_srb = sa.generate_ngrams(corpus_srb, (2, 2))
        bigrams_eng = sa.generate_ngrams(corpus_eng, (2, 2))

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                                 [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0]])
        expected_eng = np.array([[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1]])

        self.assertTrue((np.array_equal(bigrams_srb, expected_srb)) and (np.array_equal(bigrams_eng, expected_eng)))

    def test_generate_ngrams_bigrams_plus_unigrams(self):
        bigrams_unigram_srb = sa.generate_ngrams(corpus_srb, (1, 2))
        bigrams_unigram_eng = sa.generate_ngrams(corpus_eng, (1, 2))

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                                 [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0]])
        expected_eng = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]])

        self.assertTrue((np.array_equal(bigrams_unigram_srb, expected_srb)) and (np.array_equal(bigrams_unigram_eng, expected_eng)))

    def test_get_part_of_speech_words(self):
        cleaned_corpus_srb = sa.convert_to_latin(corpus_srb)
        cleaned_corpus_srb = sa.remove_punctuation(cleaned_corpus_srb, 'Serbian')
        cleaned_corpus_srb = sa.word_normalization(cleaned_corpus_srb, 'Serbian')
        pos_tag_srb = sa.get_part_of_speech_words(cleaned_corpus_srb, 'Serbian')
        pos_tag_eng = sa.get_part_of_speech_words(corpus_eng, 'English')

        expected_srb = [[('film', 'n:m'), ('predstavlja', 'n:n'), ('oda', 'n:f'), ('zxivotan', 'a:aen'), ('stilo', 'n:n'), ('jedan', 'num'), ('pacifista', 'n:m')],
                        [('the', 'int'), ('big', 'n:m'), ('lewbowski', 'a:aem'), ('ona', 'pro'), ('klasicxan', 'a:ben'), ('pricxa', 'n:f'), ('prevara', 'n:f'), ('kriminal', 'n:m'), ('i', 'conj'), ('spletkarenje', 'n:n'), ('viditi', 'v:f'), ('kroz', 'prep'), ('ocxi', 'n:f'), ('skroman', 'a:bef'), ('cxovek', 'n:m')]]

        expected_eng = [[('I', 'PRP'), ('do', 'VBP'), ("n't", 'RB'), ('like', 'VB'), ('movie', 'NN'), ("'The", 'IN'), ('Godfather', 'NNP'), ("'", 'POS'), ('at', 'IN'), ('all', 'DT'),
                         ('.', '.'), ('It', 'PRP'), ('is', 'VBZ'), ('terribly', 'RB'), ('boring', 'JJ'), (',', ','), ('long', 'JJ'), ('and', 'CC'), ('uninteresting', 'JJ'), ('!', '.')],
                        [('I', 'PRP'), ('recommend', 'VBP'), ('everyone', 'NN'), ('to', 'TO'), ('watch', 'VB'), ('the', 'DT'), ('movie', 'NN'), ("'Memento", 'POS'), ("'", 'POS'), ('.', '.'),
                         ('I', 'PRP'), ("'m", 'VBP'), ('thrilled', 'JJ'), ('with', 'IN'), ('it', 'PRP'), ('.', '.')]]

        self.assertTrue((pos_tag_srb == expected_srb) and (pos_tag_eng == expected_eng))

    def test_part_of_speech_tagging(self):
        cleaned_corpus_srb = sa.convert_to_latin(corpus_srb)
        cleaned_corpus_srb = sa.remove_punctuation(cleaned_corpus_srb, 'Serbian')
        cleaned_corpus_srb = sa.word_normalization(cleaned_corpus_srb, 'Serbian')
        pos_tag_srb = sa.part_of_speech_tagging(cleaned_corpus_srb, 'Serbian')
        pos_tag_eng = sa.part_of_speech_tagging(corpus_eng, 'English')

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1.],
                                 [1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0.]])
        expected_eng = np.array([[1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                [0., 1., 1., 0., 1., 0., 2., 0., 2., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])

        self.assertTrue((np.allclose(pos_tag_srb, expected_srb)) and (np.allclose(pos_tag_eng, expected_eng)))

    def test_get_word_position(self):
        word_position_srb = sa.get_word_position(corpus_srb)
        word_position_eng = sa.get_word_position(corpus_eng)

        expected_srb = [[('Film', 'begin'), ('predstavlja', 'begin'), ('odu', 'midle'), ('životnom', 'midle'), ('stilu', 'midle'), ('jednog', 'end'), ('pacifiste', 'end'),('.', 'end')],
                        [('The', 'begin'), ("'Big", 'begin'), ('Lewbowski', 'begin'), ("'", 'begin'), ('je', 'begin'), ('klasična', 'begin'),
                        ('priča', 'midle'), ('prevare', 'midle'), (',', 'midle'), ('kriminala', 'midle'), ('i', 'midle'), ('spletkarenja', 'midle'),
                        ('viđena', 'end'), ('kroz', 'end'), ('oči', 'end'), ('skromnog', 'end'), ('čoveka', 'end'), ('.', 'end')]]
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
        expected_srb = np.array([[0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1.],
                                 [1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 0., 1., 1., 0.]])
        expected_eng = np.array([[1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                 [0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])

        self.assertTrue((np.allclose(position_srb, expected_srb)) and (np.allclose(position_eng, expected_eng)))

    def test_create_vocabulary(self):
        tagged_position_corpus_eng = sa.get_word_position(corpus_eng)
        tagged_pos_tag_corpus_eng = sa.get_part_of_speech_words(corpus_eng, 'English')

        vocabulary_position_eng = sa.create_vocabulary(tagged_position_corpus_eng)
        vocabulary_pos_tag_eng = sa.create_vocabulary(tagged_pos_tag_corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_position_eng = [('!', 'end'), ("'", 'midle'), ("'Memento", 'midle'), ("'The", 'begin'), ("'m", 'end'), (',', 'end'), ('.', 'end'), ('.', 'midle'), ('Godfather', 'midle'), ('I', 'begin'), ('I', 'end'), ('It', 'midle'), ('all', 'midle'), ('and', 'end'), ('at', 'midle'), ('boring', 'end'), ('do', 'begin'), ('everyone', 'begin'),
                                 ('is', 'midle'), ('it', 'end'), ('like', 'begin'), ('long', 'end'), ('movie', 'begin'), ('movie', 'midle'), ("n't", 'begin'), ('recommend', 'begin'), ('terribly', 'end'), ('the', 'midle'), ('thrilled', 'end'), ('to', 'begin'), ('uninteresting', 'end'), ('watch', 'begin'), ('with', 'end')]
        expected_pos_tag_eng = [('!', '.'), ("'", 'POS'), ("'Memento", 'POS'), ("'The", 'IN'), ("'m", 'VBP'), (',', ','), ('.', '.'), ('Godfather', 'NNP'), ('I', 'PRP'), ('It', 'PRP'), ('all', 'DT'), ('and', 'CC'), ('at', 'IN'), ('boring', 'JJ'), ('do', 'VBP'), ('everyone', 'NN'),
                                ('is', 'VBZ'), ('it', 'PRP'), ('like', 'VB'), ('long', 'JJ'), ('movie', 'NN'), ("n't", 'RB'), ('recommend', 'VBP'), ('terribly', 'RB'), ('the', 'DT'), ('thrilled', 'JJ'), ('to', 'TO'), ('uninteresting', 'JJ'), ('watch', 'VB'), ('with', 'IN')]

        self.assertTrue((vocabulary_position_eng == expected_position_eng) and (vocabulary_pos_tag_eng == expected_pos_tag_eng))

    def test_create_model(self):
        tagged_position_corpus_eng = sa.get_word_position(corpus_eng)
        tagged_pos_tag_corpus_eng = sa.get_part_of_speech_words(corpus_eng, 'English')

        vocabulary_position_eng = sa.create_vocabulary(tagged_position_corpus_eng)
        vocabulary_pos_tag_eng = sa.create_vocabulary(tagged_pos_tag_corpus_eng)

        position_model_eng = sa.create_model(tagged_position_corpus_eng, vocabulary_position_eng)
        pos_tagging_model_eng = sa.create_model(tagged_pos_tag_corpus_eng, vocabulary_pos_tag_eng)

        # NOTE: arrays are in alphabetical order
        expected_position_eng = np.array([[1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                          [0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])
        expected_pos_tag_eng = np.array([[1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
                                         [0., 1., 1., 0., 1., 0., 2., 0., 2., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1.]])

        self.assertTrue((np.allclose(position_model_eng, expected_position_eng)) and (np.allclose(pos_tagging_model_eng, expected_pos_tag_eng)))

    def test_compute_tf(self):

        tf_srb = sa.compute_tf(corpus_srb)
        tf_eng = sa.compute_tf(corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0., 0., 0., 0.14285714, 0., 0., 0., 0., 0.14285714, 0., 0.14285714, 0.14285714, 0., 0., 0., 0., 0.14285714, 0., 0., 0., 0.14285714],
                                 [0., 0., 0.07142857, 0., 0.07142857, 0.07142857, 0.07142857, 0., 0., 0.07142857, 0., 0., 0.07142857, 0.07142857, 0.07142857, 0.07142857, 0., 0., 0.07142857, 0.07142857, 0.]])
        expected_eng = np.array([[0.07142857, 0.07142857, 0.07142857, 0.07142857, 0., 0., 0., 0.07142857, 0., 0.07142857, 0.07142857, 0., 0.07142857, 0., 0.07142857, 0., 0., 0., 0.07142857, 0., 0.],
                                 [0., 0., 0., 0., 0., 0.1, 0., 0., 0.1, 0., 0., 0., 0.1, 0.1, 0., 0.1, 0.1, 0.1, 0., 0.1, 0.1]])

        self.assertTrue((np.allclose(tf_srb, expected_srb)) and (np.allclose(tf_eng, expected_eng)))

    def test_compute_tf_idf(self):
        tf_idf_srb = sa.compute_tf_idf(corpus_srb)
        tf_idf_eng = sa.compute_tf_idf(corpus_eng)

        # NOTE: arrays are in alphabetical order
        expected_srb = np.array([[0., 0.37796447, 0., 0.37796447, 0., 0., 0., 0., 0.37796447, 0., 0.37796447, 0.37796447, 0., 0., 0., 0., 0.37796447, 0., 0., 0., 0.37796447],
                                 [0.26726124, 0., 0.26726124, 0., 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0., 0.26726124, 0., 0., 0.26726124, 0.26726124, 0.26726124, 0.26726124, 0., 0.26726124, 0.26726124, 0.26726124, 0.]])
        expected_eng = np.array([[0.28263102, 0.28263102, 0.28263102, 0.28263102, 0.28263102, 0., 0.28263102, 0.28263102, 0.2010943, 0.28263102, 0.28263102, 0., 0.2010943, 0., 0.28263102, 0.2010943, 0., 0., 0.28263102, 0., 0.],
                                 [0., 0., 0., 0., 0., 0.34261985, 0., 0., 0.24377685, 0., 0., 0.34261985, 0.24377685, 0.34261985, 0., 0.24377685, 0.34261985, 0.34261985, 0., 0.34261985, 0.34261985]])

        self.assertTrue((np.allclose(tf_idf_srb, expected_srb)) and (np.allclose(tf_idf_eng, expected_eng)))

if __name__ == '__main__':
    unittest.main()
