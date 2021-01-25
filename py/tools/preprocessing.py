import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# sw : stopword option
def tok(doc, sw=False):
    tokenized = nltk.word_tokenize(doc)
    lemmatizer = WordNetLemmatizer()
    if sw:
        stop_words = set(stopwords.words('english'))
        result = [lemmatizer.lemmatize(word.lower()) for word in tokenized if word not in stop_words and word.isalpha()]
    else:
        result = [lemmatizer.lemmatize(word) for word in tokenized]
    return result


def bow(doc, vectorizer=None):
    doc = tok(doc, sw=True)
    lemme = WordNetLemmatizer()
    docs = []
    for d in doc:
        li = []
        for word in d:
            li.append(lemme.lemmatize(word))
        docs.append(" ".join(li))
    if vectorizer is None:
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(docs)
    else:
        bow = vectorizer.transform(docs)
    bow = bow.toarray()
    bow = pd.DataFrame(bow)
    return bow, vectorizer
