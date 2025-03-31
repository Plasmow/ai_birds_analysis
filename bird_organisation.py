import pandas as pd
import os

# Définir les dossiers d'entrée et de sortie
input_dir = "BirdCounts_with_rendements"
output_dir = "BirdCounts_with_rendements_transformed"
os.makedirs(output_dir, exist_ok=True)

# Charger les codes des pays
country_codes = pd.read_csv('country codes.csv', sep=";", index_col=0)

# Parcourir tous les fichiers du dossier d'entrée
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):  # Vérifier que c'est un fichier CSV
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".csv", "_transformed.csv"))
        
        # Charger le fichier
        bird_counts = pd.read_csv(input_path, sep=";", decimal=",")
        
        # Remplacer les numéros de pays par les noms de pays
        bird_counts['site'] = bird_counts['site'].map(country_codes['x'])
        
        # Réorganiser le DataFrame pour avoir une colonne par pays
        bird_counts_pivot = bird_counts.pivot(index='year', columns='site', values='count')
        
        # Ajouter les colonnes de rendements pour chaque pays
        for country in bird_counts_pivot.columns:
            bird_counts_pivot[f'{country}_rendements'] = bird_counts_pivot[country].pct_change()
        
        bird_counts_pivot.dropna(how='all', inplace=True)
        
        # Sauvegarder le fichier transformé
        bird_counts_pivot.to_csv(output_path, sep=";", decimal=",")
        
        print(f"Fichier transformé : {output_path}")

print("Tous les fichiers ont été transformés avec succès.")