import glob
import re
from pandas import DataFrame
from tools.preprocessing import tok

# On itère sur chacun des documents pour extraire les différents mots présents ayant passé les
# étapes de segmentation et de lemmatisation afin de construire le dictionnaire et la matrice
# d'incidence.
columns = ["terms", "id_doc", "freq"]
df = DataFrame(columns=columns)
id_file = 1
id_line = 0

for files in glob.glob("../docs/*.txt"):
    path = files.replace("\\", "/")
    file = open(path, "r", encoding="utf8")
    # Pour chaque fichier, on va venir tokeniser l'ensemble des phrases
    num_ligne = 0
    name_file = file.name.rsplit("/", 1)[1]
    print("[FILE: ", file.name, "] tok = starting")
    for line in file:
        if line is not None:
            for words in tok(line, sw=True):
                # On convertit les majuscules
                #words = words.lower()
                if words in df.terms:
                    df[words]["freq"] += 1
                else:
                    df.loc[id_line] = [words, id_file, 1]
                    id_line += 1
            num_ligne += 1
            print("[LINE:", num_ligne, "] : ok")
    id_file += 1
    print("[FILE: ", file.name, "] tok = done")

print(df)
df.to_csv("../docs/dictionnary/dictionnary.csv")

# Création de la matrice d'incidence
