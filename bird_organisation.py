import pandas as pd
import os


# Charger les fichiers nécessaires
bird_counts = pd.read_csv('BirdCounts_with_rendements/BIRD_90_0_counts_with_rendements.csv', sep=";", decimal=",")
country_codes = pd.read_csv('country codes.csv', sep=";", index_col=0)

# Remplacer les numéros de pays par les noms de pays
bird_counts['site'] = bird_counts['site'].map(country_codes['x'])

# Réorganiser le DataFrame pour avoir une colonne par pays
bird_counts_pivot = bird_counts.pivot(index='year', columns='site', values='count')

# Ajouter les colonnes de rendements pour chaque pays
for country in bird_counts_pivot.columns:
    bird_counts_pivot[f'{country}_rendements'] = bird_counts_pivot[country].pct_change()
bird_counts_pivot.dropna(how='all', inplace=True)

# Sauvegarder le nouveau fichier CSV
output_dir = "BirdCounts_with_rendements_transformed"
os.makedirs(output_dir, exist_ok=True)
bird_counts_pivot.to_csv('BirdCounts_with_rendements_transformed/BIRD_90_0_counts_with_rendements_transformed.csv', sep=";", decimal=",")

print("Le fichier des populations d'oiseaux a été transformé avec succès.")