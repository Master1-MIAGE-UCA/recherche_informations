import glob
import re
import pandas as pd
from tools.preprocessing import preproc
from tools.request import exec_request
import numpy as np
import pickle

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
    df.to_pickle("../docs/generated_data/incidence_matrix.pkl")


def create_reverse_idx():
    # On récupère notre matrice d'incidence
    matrix = pd.read_pickle("../docs/generated_data/incidence_matrix.pkl")
    # Création du dictionnaire de mots
    dictionnary = [x for x in matrix.index]
    # Création de l'index inversé
    idx_inv = {}
    # On construit la base de notre dataframe avec les mots du dictionnaires et une liste de documents vide
    for word in dictionnary:
        idx_inv[word] = matrix.columns[np.arange(len(matrix.columns))[matrix.loc[word] > 0]].tolist()
        print("[Line '", word, "'] added")
    # On sauvegarde notre map dans un fichier .pkl tel que
    with open("../docs/generated_data/idx_inv.pkl", "wb") as f:
        pickle.dump(idx_inv, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    print("Le fichier", f.name, "a été généré\n")


print("[MAIN] lancement du programme 'main.py' : Recherche d'informations")
end_program = False
while end_program is False:
    request = input("Entrez une requete > ")
    # Arret du programme
    if request.lower() == "exit":
        print("[MAIN] Fin du programme")
        end_program = True
        break
    # Appel de notre algorithme de recherche
    exec_request(request="disease AND severe AND pneumonia")
    print("[FIN DE LA RECHERCHE]\n")
