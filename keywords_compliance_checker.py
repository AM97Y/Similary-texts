# pylint: disable=unused-argument

import gensim
import pke
from nltk.corpus import wordnet
from rake_nltk import Rake

from text_rank import TextRankKeywords
from tfidf import TfidfKeywords
from utils import remove_urls, pre_process, text2words
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")


def check_keywords_compliance(text: str,
                              keywords: set[str],
                              algorithm: str = 'SingleRank'):
    """
    Checks if a given text complies with the given set of keywords.
    :param algorithm: Algorithm for extract keywords.
    :param text: Text for check.
    :param keywords: Key words for check.
    :return: Checked or not.
    """

    text = remove_urls(text=text)
    train_text = pre_process(text)
    keywords = pre_process(' '.join(keywords)).split()
    number_keywords = len(keywords) * 10

    found_keywords = _get_keywords_by_text(train_text, algorithm, number_keywords)

    for found_keyword in found_keywords:
        if len(found_keyword.split()) > 1:
            found_keywords += found_keyword.split()
    if len(found_keywords) == 0:
        return False, []
    return _check_keywords(keywords=keywords,
                           found_keywords=found_keywords, text=text)


def _get_keywords_by_text(text: str,
                          algorithm: str,
                          number_keywords: int) -> list[str]:
    """
    This function finds keywords for text using the algorithm.
    """

    algorithms = {'TfiDF':
                      _get_keywords_tfidf,
                  'TextRank':
                      _get_keywords_text_rank,
                  'TextRankGensim':
                      _get_keywords_text_rank_gensim,
                  'Rake':
                      _get_keywords_rake,
                  'SingleRank':
                      _get_keywords_single_rank,
                  "TopicRank":
                      _get_keywords_topic_rank,
                  }
    found_keywords = algorithms.get(algorithm, [])(text, number_keywords)

    return found_keywords


def _get_keywords_tfidf(text: str, number_keywords: int) -> list[str]:
    """
    This function finds keywords using the algorithm TFIDF.
    """
    tfidf = TfidfKeywords()
    found_keywords = list(
        tfidf.get_keywords(text=text, number=number_keywords).keys())
    return found_keywords


def _get_keywords_text_rank(text: str, number_keywords: int) -> list[str]:
    """
    This function finds keywords using the algorithm Text Rank.
    """
    text_rank_model = TextRankKeywords()
    return text_rank_model.get_keywords(text,
                                        window_size=4, lower=True,
                                        number=number_keywords)


def _get_keywords_text_rank_gensim(text: str, number_keywords: int) -> list[str]:
    """
    This function finds keywords using the algorithm Text Rank by gensim.
    """
    return gensim.summarization.keywords(text,
                                         # use 50% of original text
                                         ratio=0.5,
                                         words=None,
                                         split=True,
                                         scores=False,
                                         pos_filter=('NN', 'JJ'),
                                         lemmatize=True,
                                         deacc=True)[:number_keywords]


def _get_keywords_topic_rank(text: str, number_keywords: int) -> list[str]:
    """
    This function finds keywords using the algorithm Topic Rank.
    """
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text,
                            language='en',
                            normalization="lemmas")
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(threshold=0.65, method='average')

    result = extractor.get_n_best(n=number_keywords)
    return [word[0] for word in result]


def _get_keywords_single_rank(text: str, number_keywords) -> list[str]:
    """
    This function finds keywords using the algorithm Single Rank.
    """
    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=text,
                            language='en',
                            normalization="lemmas")

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(window=10,
                                  pos={'NOUN', 'PROPN', 'ADJ'})
    result = extractor.get_n_best(n=number_keywords)
    return [word[0] for word in result]


def _get_keywords_rake(text: str, number_keywords: int) -> list[str]:
    """
    This function finds keywords using the algorithm Rake.
    """
    rake_model = Rake()
    rake_model.extract_keywords_from_text(text)
    return \
        list(rake_model.get_word_frequency_distribution().keys())[:number_keywords]


def _get_synonyms_word(word: str) -> set[str]:
    """
    This function returns synonyms for the word.
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name()
                            .replace('_', ' ')
                            .replace('-', ' '))
    return set(synonyms)


def _get_synonyms(words: set[str]) -> set[str]:
    """
    This function return synonyms for words.
    """
    synonyms = set()

    for word in words:
        synonyms.union(_get_synonyms_word(word))

    return synonyms


def _get_antonyms_word(word: str) -> set[str]:
    """
    This function returns antonyms for the word.
    """
    antonyms = []

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                antonyms.append(
                    lemma.antonyms()[0].name()
                        .replace('_', ' ')
                        .replace('-', ' '))
    return set()


def _get_antonyms(words: set[str]) -> set[str]:
    """
    This function return antonyms for words.
    """
    antonyms = set()
    for word in words:
        antonyms.union(_get_antonyms_word(word))
    return antonyms


def _check_antonyms(words: set[str], keyword: str) -> bool:
    """
    This function checks that the given keyword is antonym by words.
    """
    return keyword in _get_antonyms(words)


def _check_synonyms(words: set[str], keyword: str) -> bool:
    """
    This function checks that the given keyword is synonym by words.
    """
    return keyword in _get_synonyms(words)


def _check_vocab(words: set[str]) -> bool:
    """
    This function checks the presence of all words in the model.
    """
    vocab_vectors = model.vocab.keys()
    for word in words:
        for i in word.split():
            if not i in vocab_vectors:
                return False
    return True


def _check_keywords(found_keywords: set[str],
                    keywords: set[str], text: str,
                    use_word2word_similarity: bool = False,
                    th_similarity: float = 0.60):
    """
    This function checks if the found keywords match the keywords.
    found_keywords:  Found keywords in the text.
    keywords: Keywords to check.
    text: All text.
    th_similarity:
    use_word2word_similarity: Check each similarity found keyword with keyword.

    return: Whether it matches keywords or not.
    """

    count = 0
    metric = []
    text_words = text2words(text)

    keywords = [word for word in keywords
                if _check_vocab(words=[word])]
    found_keywords = [word for word in found_keywords
                      if _check_vocab(words=[word])]
    text_words = [word for word in text_words
                  if _check_vocab(words=[word])]
    if not keywords:
        return False, metric

    for keyword in keywords:
        if _check_antonyms(found_keywords, keyword):
            continue
        if keyword in found_keywords or _check_synonyms(words=found_keywords,
                                                        keyword=keyword):
            count += 1
            continue

        if model.n_similarity(text_words, keyword.split()) >= th_similarity:
            metric.append(model.n_similarity(text_words, keyword.split()))
            count += 1

        elif use_word2word_similarity:
            count += _check_similarity_word2words(keyword=keyword,
                                                  found_keywords=found_keywords,
                                                  th_similarity=th_similarity)

        print('_____')
        print(count >= len(keywords), metric, count)

    return count >= len(keywords), metric


def _check_similarity_word2words(keyword: str, found_keywords: set[str],
                                 th_similarity: float) -> int:
    """
    This function compares the similarity of a word with each found separately.
    """
    for i in found_keywords:
        if model.n_similarity(i.split(), keyword.split()) >= th_similarity:
            return 0
    return 1
