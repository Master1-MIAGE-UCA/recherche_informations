import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
from preprocessing import preproc

requetes = ["desease AND severe AND pneumonia", "antibody AND plasma AND (cells OR receptors)",
            "antimalarial drugs OR antiviral agents OR immunomodulators",
            "NOT plasma AND risk of infection AND restrictions",
            "(older adults AND antibodies) AND NOT (genomes OR variant)"]

keywords_bool_request = ["AND", "OR", "NOT"]

preprocess_overall = []


def request(req):
    # On teste la requete pour identifier quel algorithme appelé en fonction de son contenu
    # Si on repère un mot clé propre aux requetes booléennes, on appelle l'algorithme dédié et vice et versa
    for word in req:
        if word in keywords_bool_request:
            return boolean_request(req)
    return complex_request(req)


def boolean_request(req):
    print("[REQUEST] req = ", req)
    preprocess_req = []
    tokenized = nltk.word_tokenize(req)
    lemmatizer = WordNetLemmatizer()
    for word in tokenized:
        if word.isupper() is True:
            preprocess_req.append(word)
        else:
            preprocess_req.append(lemmatizer.lemmatize(word))
    return preprocess_req


def complex_request(req):
    # On charge notre index inversé
    idx_inv = pd.read_pickle("../../docs/generated_data/idx_inv.pkl")
    # On vient appliquer notre algorithme 'preproc' sur la requete
    list_words = preproc(req)
    # On itère sur chacun des mots et on vient récupérer leur liste de documents respectives


# test
for r in requetes:
    preprocess_overall.append(boolean_request(r))

print("[END] preprocess_overall: ", preprocess_overall, "\n")

complex_request("efficacy and safety of the treatments")
