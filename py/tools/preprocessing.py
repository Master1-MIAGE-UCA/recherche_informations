import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# sw : stopword option
def preproc(doc, sw=False):
    tokenized = nltk.word_tokenize(doc)
    lemmatizer = WordNetLemmatizer()
    if sw:
        stop_words = set(stopwords.words('english'))
        result = [lemmatizer.lemmatize(word.lower()) for word in tokenized if word not in stop_words and word.isalpha()]
    else:
        result = [lemmatizer.lemmatize(word) for word in tokenized]
    return result
