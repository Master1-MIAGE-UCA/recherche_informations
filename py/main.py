import glob
import re
from tools.preprocessing import tok

# On itère sur chacun des documents pour extraire les différents mots présents ayant passé les
# étapes de segmentation et de lemmatisation afin de construire le dictionnaire et la matrice
# d'incidence.
# doc_tokenized = []
for files in glob.glob("../docs/*.txt"):
    path = files.replace("\\", "/")
    file = open(path, "r")
    # Pour chaque fichier, on va venir tokeniser l'ensemble des phrases
    num_ligne = 0
    name_file = file.name.rsplit("/", 1)[1]
    newfile = open("../docs/tokenized_files/{}".format(name_file), "w")
    for line in file:
        if line is not None:
            print("\n[FILE: ", name_file, "] : starting process ...")
            for words in tok(line, sw=True):
                newfile.write(str(words) + ";")
                print("[LINE:", num_ligne, "] : writed")
                num_ligne += 1
                # doc_tokenized.append(word)
    newfile.close()
    print("[FILE: ", file.name, "] tok = done\n")
