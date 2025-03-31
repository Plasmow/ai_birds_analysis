import pandas as pd
import os

# Charger le fichier CSV dans un DataFrame

file_path = "Pressure data/FARMSIZE.csv"
file_path = "Pressure data/URB.csv"
df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)

# Vérifier les premières lignes pour avoir un aperçu des données
print("Fichier chargé :")
print(df.head())

# Calculer la différence relative pour chaque ligne et chaque colonne
for i in range(2, len(df.columns)):  # Commence à l'indice 2 (colonne C)
    col = df.columns[i]  # Colonne actuelle (par exemple, C, E, G, etc.)
    
    # Calcul de la différence relative : (valeur actuelle - valeur précédente) / valeur précédente
    df[f'{col}_rendements'] = df[col].pct_change()  # pct_change() calcule la différence relative

# Afficher les nouvelles colonnes pour vérifier
print("\nDonnées après ajout des colonnes de rendements :")
print(df.head())

# Afficher le répertoire actuel
print("Répertoire actuel avant changement :", os.getcwd())

# Changer le répertoire de travail
os.chdir('Pressure data')

# Vérifier que le répertoire a bien été changé
print("Répertoire actuel après changement :", os.getcwd())

# Sauvegarder les résultats dans un nouveau fichier CSV
output_file = 'URB_with_rendements.csv'
df.to_csv(output_file, index=False)

print(f"Fichier sauvegardé avec succès sous {output_file}.")
