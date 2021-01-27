import glob
import re
import pandas as pd
from tools.preprocessing import tok

# On itère sur chacun des documents pour extraire les différents mots présents ayant passé les
# étapes de segmentation et de lemmatisation afin de construire le dictionnaire et la matrice
# d'incidence.
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
            for words in tok(line, sw=True):
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

print(df)
df.to_csv("../docs/generated_data/matrix.csv")

# Création de l'index inversé
idx_inv = pd.DataFrame(columns=["list_docs"])

# Pour ce faire, nous allons itérer sur le dataframe correspondant à la matrice d'incidence générée
# un peu plus tôt
# todo
