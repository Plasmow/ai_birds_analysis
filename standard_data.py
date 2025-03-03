import pandas as pd
import os

# Charger le fichier CSV dans un DataFrame
file_path = "Pressure data/FARMSIZE.csv"
df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)

# Afficher les premières lignes pour vérifier
print(df.head())

# Calculer la différence relative entre les colonnes (en commençant par C)
for i in range(2, len(df.columns)-1, 2):  # Commence à l'indice 2 (colonne C)
    col1 = df.columns[i]     # Par exemple 'C'
    col2 = df.columns[i+1]   # Par exemple 'E'
    
    # Calcul de la différence relative et ajout dans une nouvelle colonne
    new_col_name = f'{col2}_{col1}_rendements'  # Nouveau nom pour la colonne calculée
    df[new_col_name] = (df[col2] - df[col1]) / df[col1]

# Afficher le DataFrame avec les nouvelles colonnes pour vérifier
print(df.head())

# Afficher le répertoire actuel
print("Répertoire actuel avant changement :", os.getcwd())

# Changer le répertoire de travail
os.chdir('/Users/r/Documents/projetS6/ai_birds_analysis/Pressure data')

# Vérifier que le répertoire a bien été changé
print("Répertoire actuel après changement :", os.getcwd())

# Sauvegarder les résultats dans un nouveau fichier CSV
df.to_csv('FARMSIZE_with_rendements.csv', index=False)

# Si vous souhaitez écraser le fichier existant, utilisez :
df.to_csv('FARMSIZE_with_rendements.csv', index=False)