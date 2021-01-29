import nltk
import re
import pickle
from nltk.stem import WordNetLemmatizer
import pandas as pd
from tools import preprocessing as proc

requetes = ["disease AND severe AND pneumonia", "antibody AND plasma AND (cells OR receptors)",
            "antimalarial drugs OR antiviral agents OR immunomodulators",
            "NOT plasma AND risk of infection AND restrictions",
            "(older adults AND antibodies) AND NOT (genomes OR variant)"]

keywords_bool_request = ["AND", "OR", "NOT"]
# Création de regex dédié à la recherche des parenthèses et des mots clés pour les requetes booléennes
rgx_parenthesis = re.compile(r"\(\s+([^()]+)\s+\)")
# rgx_and = re.compile(r"AND\s+(\S+)")

preprocess_overall = []


def exec_request(request: str):
    # On teste la requete pour identifier quel algorithme appelé en fonction de son contenu
    # Si on repère un mot clé propre aux requetes booléennes, on appelle l'algorithme dédié et vice et versa
    for word in request.split(" "):
        if word in keywords_bool_request:
            return boolean_request(request)
    return complex_request(request)


def boolean_request(req):
    if req == "(older adults AND antibodies) AND NOT (genomes OR variant)":
        print("[DEV] Cette requete renvoie une erreur dans notre algo et n'est donc pas traité pour le moment")
        return
    print("[REQUETE] exécution de la requete: ", req)
    # On applique le preprocess sur nos requetes
    req = ' '.join(proc.preproc(req, sw=False))
    # On sépare nos caractère en une liste avec comme séparateur les mots clés "AND"
    req = req.split(" AND ")
    # On récupère notre matrice d'incidence
    inc_matrix: pd.DataFrame = pd.read_pickle("../docs/generated_data/incidence_matrix.pkl")
    # On définit nos sous-résultats
    subdfs = []
    for elem in req:
        # On ajoute les statistiques recueillies depuis la matrice d'incidence pour le document courant
        if "NOT" in elem:
            real_name = elem.split("NOT ")[1]
            subdfs_cp = inc_matrix.loc[inc_matrix.index == real_name].copy()
            # On définit les valeurs de cette ligne à -1 pour les valeurs positives 0 sinon
            # pour simplifier le tri par la suite
            values: list = subdfs_cp.loc[real_name]
            for i in range(0, len(values)):
                values[i] = 0 if values[i] == 0 else -1
            subdfs_cp.loc[real_name] = values
            subdfs.append(subdfs_cp)
        else:
            subdfs.append(inc_matrix.loc[inc_matrix.index == elem])
    # On merge nos résultats
    result = pd.concat(subdfs)
    # On cherche dans le tableau si on a une valeur égale à 0 alors on supprime la colonne
    for column in result.columns:
        for idx in result.index:
            if result.loc[idx][column] == 0 or result.loc[idx][column] == -1:
                result.pop(column)
                break
    # Si notre dataset de résultat est vide, on signale à l'utilisateur qu'aucun document n'a été trouvé pour cette
    # requete
    if result.empty:
        print("[REQUETE] Aucun document n'a été trouvé pour cette requete")
    else:
        # Sinon on mesure les tf-idf et on termine notre traitement
        # Calcul du TF-IDF
        tfidf = proc.tfidf(result)
        list_tfidf = list(reversed(tfidf.argsort()))
        print("[REQUETE] Résultat final (par ordre de pertinence):")
        for idx in list_tfidf:
            print("[DOC] {} - TF-IDF={}".format(result.columns[idx], tfidf[idx]))


def complex_request(req):
    print("[REQUETE] exécution de la requete: ", req)
    # On applique le preprocess sur nos requetes
    req = ' '.join(proc.preproc(req, sw=True))
    # On sépare nos caractère en une liste avec comme séparateur les mots clés "AND"
    req = req.split(" ")
    # On récupère notre matrice d'incidence
    inc_matrix: pd.DataFrame = pd.read_pickle("../docs/generated_data/incidence_matrix.pkl")
    # On lit notre requete et on filtre notre dataset en fonction des instructions
    subdfs = []
    for elem in req:
        subdfs.append(inc_matrix.loc[inc_matrix.index == elem])
    result = pd.concat(subdfs)
    # On cherche dans le tableau si on a une valeur égale à 0 alors on supprime la colonne
    for column in result.columns:
        for idx in result.index:
            if result.loc[idx][column] == 0:
                result.pop(column)
                break
    # Calcul du TF-IDF
    tfidf = proc.tfidf(result)
    list_tfidf = list(reversed(tfidf.argsort()))
    print("[REQUETE] Résultat final (par ordre de pertinence):")
    for idx in list_tfidf:
        print("[DOC] {} - TF-IDF={}".format(result.columns[idx], tfidf[idx]))
