import pandas as pd
import os

# Définir les répertoires d'entrée et de sortie
input_dir = 'Pressure data'
output_dir = 'Pressure with rendements'

# Créer le répertoire de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Itérer sur tous les fichiers dans le répertoire d'entrée
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_dir, filename)
        
        # Charger le fichier CSV dans un DataFrame
        df = pd.read_csv(file_path, sep=";", decimal=",", index_col=0)
        
        # Calculer la différence relative pour chaque ligne et chaque colonne
        for i in range(1, len(df.columns)):  # Commence à l'indice 1 (colonne B)
            col = df.columns[i]  # Colonne actuelle (par exemple, B, C, D, etc.)
            
            # Calcul de la différence relative : (valeur actuelle - valeur précédente) / valeur précédente
            df[f'{col}_rendements'] = df[col].pct_change()  # pct_change() calcule la différence relative
        
        # Sauvegarder les résultats dans un nouveau fichier CSV en conservant la première ligne
        output_filename = f"{os.path.splitext(filename)[0]}_with_rendements.csv"
        df.to_csv(os.path.join(output_dir, output_filename), sep=";", decimal=",", index=True)

print("Tous les fichiers ont été traités et sauvegardés avec les rendements.")