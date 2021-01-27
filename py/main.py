import glob
import re
import pandas as pd
from tools.preprocessing import preproc


# On itère sur chacun des documents pour extraire les différents mots présents ayant passé les
# étapes de segmentation et de lemmatisation afin de construire le dictionnaire et la matrice
# d'incidence.
def create_incidence_matrix():
    columns = []
    id_file = 0
    # Ajout des colonnes du tableau
    for files in glob.glob("../docs/*.txt"):
        path = files.replace("\\", "/")
        file = open(path, "r", encoding="utf8")
        name_file = file.name.rsplit("/", 1)[1]
        columns.append(name_file)
        file.close()
    # Création du dataframe
    df = pd.DataFrame(columns=columns)
    # Création de la matrice d'incidence
    for files in glob.glob("../docs/*.txt"):
        path = files.replace("\\", "/")
        file = open(path, "r", encoding="utf8")
        # On récupère le nom du fichier courant
        name_file = columns[id_file]
        # Log pour tracer l'ensemble du traitement
        num_ligne = 0
        print("[FILE: ", name_file, "] tok = starting")
        # Pour chaque fichier, on va venir tokeniser l'ensemble des phrases
        for line in file:
            # Si la ligne n'est pas vide
            if line is not None:
                for words in preproc(line, sw=True):
                    # On convertit les majuscules
                    if words in df.index:
                        # On incrémente si on trouve plusieurs fois un mot dans un document
                        df.loc[words][name_file] += 1
                    else:
                        # On définit les valeurs à 0 pour toutes les colonnes
                        df.loc[words] = [0] * len(columns)
                        # On va chercher la colonne correspondante au document courant pour définir
                        # la valeur de la casee à 1
                        df.loc[words][name_file] = 1
                num_ligne += 1
                print("[LINE:", num_ligne, "] : ok")
        print("[FILE: ", file.name, "] tok = done\n")
        id_file += 1
        file.close()
    # On sauvegarde l'instance de notre dataframe dans un fichier .pkl
    df.to_pickle("../docs/generated_data/matrix.pkl")


def create_reverse_idx():
    # On récupère notre matrice d'incidence
    matrix = pd.read_pickle("../docs/generated_data/matrix.pkl")
    # Création du dictionnaire de mots
    dictionnary = [x for x in matrix.index]
    # Création de l'index inversé
    idx_inv = pd.DataFrame(columns=["list_docs"])
    # On construit la base de notre dataframe avec les mots du dictionnaires et une liste de documents vide
    for word in dictionnary:
        idx_inv.loc[word] = [[]]
        print("[Line '", word, "'] added")
    # On itère dans chaque élement de notre tableau et on va analyser en parallèle les lignes de notre
    # matrice d'incidence
    for idx in idx_inv.index:
        # On fait la jointure entre la ligne du tableau et notre matrice
        values = matrix.loc[idx]
        # print(values)
        # Pour chaques valeurs non nulles, on vient écrire le nom de la colonne (qui correspond au nom du fichier)
        # dans la ligne correspondant au mot courant de notre index inversé
        for doc_name in values.index:
            if values.loc[doc_name] != 0:
                idx_inv.loc[idx]["list_docs"].append(doc_name)
        print("[Idx: '", idx, "'] done")
    # On sauvegarde l'instance de notre dataframe dans un fichier .pkl
    idx_inv.to_pickle("../docs/generated_data/idx_inv.pkl")


print("[MAIN] lancement du programme 'main.py' : Recherche d'informations")
end_program = False
while end_program is False:
    request = input("Entrez une requete > ")
    # Appel de notre algorithme de recherche

    # Arret du programme
    if request.lower() == "exit":
        print("[MAIN] Fin du programme")
        end_program = True
