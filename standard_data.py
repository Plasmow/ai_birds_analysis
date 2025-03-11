import pandas as pd
import numpy as np
import os

file_path_example = "Pressure data/FARMSIZE.csv"
data={}

def load_data(file_path):
    df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)



    df = df.apply(pd.to_numeric, errors='coerce')

    # Calcul du rendement pour chaque pays en suivant la formule donnée
    # Utilisation de shift(1) pour obtenir les valeurs de la colonne suivante (t+1)
    df_normalized = (df.shift(-1) - df) / df

    # Remplacement des valeurs extrêmes par NaN (par exemple, valeurs infinies ou NaN obtenues dans le calcul)
    df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)
    
    
directory='Pressure data'

def get_all_data(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            data[file]= os.path.join(root, file)

get_all_data(directory)

print(data['PLA.csv'])
