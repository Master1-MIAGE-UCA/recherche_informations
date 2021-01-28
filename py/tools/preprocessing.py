import glob
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer


# sw : stopword option
def preproc(doc, sw=True):
    tokenized = nltk.word_tokenize(doc)
    lemmatizer = WordNetLemmatizer()
    if sw:
        stop_words = stopwords.words('english')
        result = [lemmatizer.lemmatize(word.lower()) for word in tokenized if word not in stop_words and word.isalpha()]
    else:
        result = [lemmatizer.lemmatize(word) for word in tokenized]
    return result


def tfidf(dataframe):
    tfIdfVectorizer = TfidfTransformer()
    tfIdf = tfIdfVectorizer.fit_transform(dataframe)
    return tfIdf.toarray().mean(axis=0)
