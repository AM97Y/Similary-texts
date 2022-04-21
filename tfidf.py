import string

import nltk
import numpy
import scipy
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


class LemmaTokenizer:  # pylint: disable=too-few-public-methods
    """
    Words tokenizer with preprocess text.
    """

    def __init__(self):
        self._wnl = WordNetLemmatizer()
        self._pos_tags = ['NN',
                          'NNS',
                          'NNP',
                          'NNPS',
                          'VBN',
                          'VBG',
                          'VBD',
                          'VBP',
                          'VBZ',
                          'VB']

    def __call__(self, words: str) -> list:
        return [self._wnl.lemmatize(i[0]) for i in nltk.pos_tag(word_tokenize(words))
                if i[0].lower() not in string.punctuation
                and i[0] not in stopwords.words('english')
                and i[1] in self._pos_tags]


class TfidfKeywords:  # pylint: disable=too-few-public-methods
    """
    Extract keywords from text by tfidf.
    """

    def get_keywords(self, text: str, number: int = 10) -> bool:
        """
        This function checks if the text matches the keywords.
        :param number: Numer of keywords.
        :param text: Text for searching keywords.
        :return: Found keywords.
        """
        vectorizer = CountVectorizer()
        # vectorizer = CountVectorizer(tokenizer=self.lt)
        word_count_vector = vectorizer.fit_transform(text.split())

        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(word_count_vector)

        feature_names = vectorizer.get_feature_names_out()
        tf_idf_vector = tfidf_transformer.transform(vectorizer.transform([text]))
        sorted_items = self._sort_coo(tf_idf_vector.tocoo())
        found_keywords = self._extract_top_keywords(feature_names=feature_names,
                                                    sorted_items=sorted_items,
                                                    number=number)

        return found_keywords

    @staticmethod
    def _sort_coo(coo_matrix: scipy.sparse.coo_matrix) -> list:
        """
        Sort the tf-idf vectors by descending order of scores.
        :param coo_matrix: A sparse matrix in COOrdinate format.
        :return: List of words and their scores.
        """
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @staticmethod
    def _extract_top_keywords(feature_names: numpy.ndarray,
                              sorted_items: list[tuple[int, int]],
                              number: int = 10) -> dict:
        """
        Get the feature names and tf-idf score of top n items.

        :param number: Top number of keywords.
        :param feature_names: List of features name.
        :param sorted_items: Sorted items by func sort_coo.
        :return: Keywords.
        """
        sorted_items = sorted_items[:number]

        score_vals = []
        feature_vals = []

        for i, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[i])

        results = {}
        for i, elem in enumerate(feature_vals):
            results[elem] = score_vals[i]
        return results
