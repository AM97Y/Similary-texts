from collections import OrderedDict

import numpy as np
import spacy


# python -m spacy download en_core_web_sm


class TextRankKeywords:  # pylint: disable=too-few-public-methods

    """
    Extract keywords from text by text rank.
    """

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self._d = 0.85  # damping coefficient, usually is .85
        self._min_diff = 1e-15  # convergence threshold
        self._steps = 20  # iteration steps
        self._node_weight = None  # save keywords and its weight
        self._pos_tags = ['NOUN', 'PROPN', 'VERB']
        self._node_weight = {}

    def get_keywords(self, text: str,
                     window_size: int = 4, lower: bool = True,
                     number: int = 10) -> list[str]:
        """
        Print top number keywords
        :param text: Text for searching keywords.
        :param lower: Lower case or not.
        :param window_size: Window size for TextRank.
        :param candidate_pos: Part of speach.
        :param number: List of top keywords.
        :return: Found keywords.
        """
        self._analyze(text, candidate_pos=self._pos_tags,
                      window_size=window_size, lower=lower)

        node_weight = OrderedDict(sorted(self._node_weight.items(),
                                         key=lambda t: t[1], reverse=True))

        return [key for i, (key, value) in enumerate(node_weight.items())
                if i < number]

    def _sentence_segment(self, doc, candidate_pos, lower):
        """
        Store those words only in cadidate_pos. Filter sentences.
        :param doc: Text for analize.
        :param candidate_pos: Part of speach for chose words.
        :param lower: Lower word case or not.
        :return: Sentences.
        """
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                if token.pos_ in candidate_pos and token.is_stop is False:
                    selected_words.append(self._text_lower(token, lower))

            sentences.append(selected_words)
        return sentences

    @staticmethod
    def _text_lower(token: str, lower: bool) -> str:
        """
        Raising text case by flag.
        """
        if lower:
            token = token.text.lower()

        return token

    @staticmethod
    def _get_vocab(sentences):
        """
        Build vocabulary.
        :param sentences: Text.
        :return: Vocab.
        """
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def _get_token_pairs(self, window_size, sentences):
        """
        Build token_pairs from windows in sentences.
        :param window_size: Window size for TextRank.
        :param sentences: Sentences by text.
        :return: Token pairs.
        """
        token_pairs = []
        for sentence in sentences:
            token_pairs += self._get_token_pairs_sentence(sentence=sentence,
                                                          window_size=window_size)
        return token_pairs

    @staticmethod
    def _get_token_pairs_sentence(sentence, window_size):
        token_pairs = []
        for i, word in enumerate(sentence):
            for j in range(i + 1, i + window_size):
                if j >= len(sentence):
                    break
                pair = (word, sentence[j])
                if pair not in token_pairs:
                    token_pairs.append(pair)
        return token_pairs

    @staticmethod
    def _symmetrize(matrix):
        """
        Get symmeric matrix.
        :param matrix: Matrix.
        :return: Symmeric matrix.
        """
        return matrix + matrix.T - np.diag(matrix.diagonal())

    def _get_matrix(self, vocab, token_pairs):
        """
        Get normalized matrix.
        :param vocab: Vocabulary.
        :param token_pairs: Token pairs.
        :return: Matrix.
        """
        # Build matrix.
        vocab_size = len(vocab)
        matrix_g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            matrix_g[i][j] = 1

        matrix_g = self._symmetrize(matrix_g)

        # Normalize matrix by column.
        norm = np.sum(matrix_g, axis=0)
        g_norm = np.divide(matrix_g, norm, where=norm != 0)

        return g_norm

    def _analyze(self, text,
                 candidate_pos,
                 window_size, lower):
        """
        Main function to analyze text.
        :param text: Text foe searching keywords.
        :param window_size: Window size for TextRank.
        :param candidate_pos: Parch of speach.
        :param lower: Lower case or not.
        """

        # Pare text by spaCy.
        doc = self.nlp(text)

        # List of list of words.
        sentences = self._sentence_segment(doc, candidate_pos, lower)
        vocab = self._get_vocab(sentences)
        token_pairs = self._get_token_pairs(window_size, sentences)
        matrix_g = self._get_matrix(vocab, token_pairs)

        # Initionlization for weight/ pagerank value.
        pagerank_weights = np.array([1] * len(vocab))

        previous_pagerank_weight = 0
        for _ in range(self._steps):
            pagerank_weights = (1 - self._d) + self._d * np.dot(matrix_g,
                                                                pagerank_weights)
            if abs(previous_pagerank_weight - sum(pagerank_weights)) \
                    < self._min_diff:
                break
            previous_pagerank_weight = sum(pagerank_weights)

        # Get weight for each node.

        for word, index in vocab.items():
            self._node_weight[word] = pagerank_weights[index]
