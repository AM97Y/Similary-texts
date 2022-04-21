import re

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def remove_urls(text: str) -> str:
    """
    This function remove https.
    """
    return re.sub(r"http://\S+|https://\S+", "", text)


def pre_process(text: str) -> str:
    """
    This function preprocess text: remove stop words,
    punctuation, lower and lemmatize.
    """
    stop_words = set(stopwords.words('english'))
    new_words = ["fig", "figure", "image", "sample", "using",
                 "show", "result", "large",
                 "also", "one", "two", "three",
                 "four", "five", "seven", "eight", "nine"]
    stop_words = list(stop_words.union(new_words))

    text = text.lower()
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.replace('[', '').replace(']', '')

    text = text.split()
    text = [word for word in text if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]

    return ' '.join(text)


def text2words(text: str) -> list[str]:
    """
    This function returns an array with the words for the text.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text_words = text.lower().split()
    return text_words
