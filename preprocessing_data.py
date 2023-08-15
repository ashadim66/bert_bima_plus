import re
from nltk.tokenize import RegexpTokenizer
# Mengubah kalimat ke kata dasar menggunakan sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd


normalizad_word = pd.read_csv(f"data/normalisasi.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

regexp = RegexpTokenizer('\w+')
txt_stopword = pd.read_csv(f"data/stopword_bima.txt",
                           names=["stopwords"], header=None)
stopword_full = set(txt_stopword.stopwords.values)


def cleaning(text):
    # Remove URLs (https/http) from review_text
    text = re.sub(r'https?\S+', ' ', text)
    # Remove mentions
    text = re.sub(r'@\S+', ' ', text)
    # Remove hashtags
    text = re.sub(r'#\S+', ' ', text)
    # Remove next character
    text = re.sub(r'\'\w+', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    return text.strip()  # Remove leading/trailing whitespace

# remove stopword pada list token


def stopwords_removal(words):
    return [word for word in words if word not in stopword_full]


def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


def bima_preprocess(review_text):
    result = review_text.lower()  # mengubah tesk ke huruf kecil
    result = cleaning(result)  # membersihkan teks
    result = regexp.tokenize(result)  # memisahkan teks
    result = stopwords_removal(result)  # membuang stopword
    result = " ".join(x for x in result)  # menggabungkan teks
    result = stemmer.stem(result)  # melakukkan stemming
    result = regexp.tokenize(result)  # memisahkan teks
    result = normalized_term(result)  # menormalisasi teks
    result = " ".join(x for x in result)  # menggabungkan teks

    return result
