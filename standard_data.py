import pandas as pd
import numpy as np

# Charger le fichier CSV en spécifiant le bon séparateur (point-virgule)
file_path = "Pressure data/FARMSIZE.csv"
df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)



df = df.apply(pd.to_numeric, errors='coerce')

# Calcul du rendement pour chaque pays en suivant la formule donnée
# Utilisation de shift(1) pour obtenir les valeurs de la colonne suivante (t+1)
df_normalized = (df.shift(-1) - df) / df

# Remplacement des valeurs extrêmes par NaN (par exemple, valeurs infinies ou NaN obtenues dans le calcul)
df_normalized = df_normalized.replace([np.inf, -np.inf], np.nan)




print(df_normalized)