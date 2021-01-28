import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


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


corpus = []
doc1=open("docs/1.txt").read()
corpus.append(doc1)
doc10=open("docs/10.txt").read()
corpus.append(doc10)
doc2=open("docs/2.txt").read()
corpus.append(doc2)
doc3=open("docs/3.txt").read()
corpus.append(doc3)
doc4=open("docs/4.txt").read()
corpus.append(doc4)
doc5=open("docs/5.txt").read()
corpus.append(doc5)
doc6=open("docs/6.txt").read()
corpus.append(doc6)
doc7=open("docs/7.txt").read()
corpus.append(doc7)
doc8=open("docs/8.txt").read()
corpus.append(doc8)
doc9=open("docs/9.txt").read()
corpus.append(doc9)

print(corpus)
tfIdfVectorizer=TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(corpus)
df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)