import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# Dossiers contenant les fichiers de données oiseaux et pressions
bird_folder = 'BirdCounts_with_rendements_transformed'
pressure_folder = 'Pressure_with_rendements'

# Parcours de chaque fichier d'oiseau
for bird_file in os.listdir(bird_folder):
    bird_path = os.path.join(bird_folder, bird_file)  
    bird_df = pd.read_csv(bird_path, delimiter=';')
    # Parcours de chaque fichier de pression
    for pressure_file in os.listdir(pressure_folder):
        pressure_path = os.path.join(pressure_folder, pressure_file)
        pressure_df = pd.read_csv(pressure_path, delimiter=';')
        # Fusion des données oiseaux et pression sur la colonne 'year'
        merged_df = bird_df.merge(pressure_df, on='year', how='inner',suffixes=("","_pressure"))

        X = []
        Y = []
        # Recherche des colonnes de rendements dans le DataFrame fusionné
        for col_bird in merged_df.columns:
            if col_bird.endswith("rendements"):
                # Recherche de la colonne de pression associée à ce rendement
                for col_pressure in merged_df.columns:
                    if col_pressure == col_bird+"_pressure": 
                        # Sélection et nettoyage des données
                        merged_df_cleaned = merged_df.loc[:, [col_bird]+[col_pressure]]
                        merged_df_cleaned = merged_df_cleaned.dropna()
                        merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "0,0").all(axis=1)]
                        merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "inf").all(axis=1)]
                        if not merged_df_cleaned.empty:
                            # Conversion des chaînes en float
                            oiseau = merged_df_cleaned[col_bird].str.replace(',', '.').astype(float)
                            pression = merged_df_cleaned[col_pressure].str.replace(',', '.').astype(float)
                            # Ajout des données à X et Y
                            Y+=oiseau.tolist()
                            X+=pression.tolist()
                            
        # Régression linéaire si données présentes
        if not(len(Y)==0):
            X = np.array(X).reshape(-1, 1)
            Y = np.array(Y)
            model = LinearRegression().fit(X, Y)

        # Affichage des résultats pour chaque couple oiseau/pression
        print("Pour l'oiseau numero ",bird_file.replace("BIRD_","").replace("_0_counts_with_rendements_transformed.csv",""))
        print("Pour la pression",pressure_file.replace("_with_rendements","").replace(".csv",""))
        print("coefficient de la pression :",model.coef_)
        print("Intercept:", model.intercept_)
        r2_score = model.score(X, Y)
        print("R² (coefficient de détermination) :", r2_score)