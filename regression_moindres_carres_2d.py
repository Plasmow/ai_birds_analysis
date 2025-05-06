import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# Charger les fichiers CSV
#farm_df = pd.read_csv('Pressure_with_rendements/FARMSIZE_with_rendements.csv', delimiter=';')
#bird_df = pd.read_csv('BirdCounts_with_rendements_transformed/BIRD_10010_0_counts_with_rendements_transformed.csv', delimiter=';')
bird_folder = 'BirdCounts_with_rendements_transformed'
pressure_folder = 'Pressure_with_rendements'
for bird_file in os.listdir(bird_folder):
    bird_path = os.path.join(bird_folder, bird_file)  # Chemin complet du fichier
    bird_df = pd.read_csv(bird_path, delimiter=';')
    for pressure_file in os.listdir(pressure_folder):
        pressure_path = os.path.join(pressure_folder, pressure_file)
        pressure_df = pd.read_csv(pressure_path, delimiter=';')
        merged_df = bird_df.merge(pressure_df, on='year', how='inner',suffixes=("","_pressure"))

        X = []
        Y = []
        for col_bird in merged_df.columns:
            if col_bird.endswith("rendements"):
                for col_pressure in merged_df.columns:
                    if col_pressure == col_bird+"_pressure": 
                        merged_df_cleaned = merged_df.loc[:, [col_bird]+[col_pressure]]
                        #merged_df_cleaned = merged_df.loc[:, [col_bird] + cols_pressure]
                        #print(merged_df_cleaned)
                        merged_df_cleaned = merged_df_cleaned.dropna()
                        merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "0,0").all(axis=1)]
                        merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "inf").all(axis=1)]
                    
                #print(merged_df_cleaned[col_bird])
                        if not merged_df_cleaned.empty:
                    
                            oiseau = merged_df_cleaned[col_bird].str.replace(',', '.').astype(float)
                    #cols_pressure_cleaned = []
                    #for col_pressure in cols_pressure:    #[col for col in merged_df_cleaned.columns if col != col_bird]:
                        #if col_pressure == col_bird+"_" + pressure_file:
                            #print(col_pressure)
                            #print(merged_df_cleaned[col_pressure])
                            #col_pressure_cleaned = merged_df_cleaned[[col_pressure]].apply(lambda x: x.str.replace(',', '.')).astype(float)
                            #print(merged_df_cleaned[col_pressure])
                            #print(merged_df_cleaned[col_pressure].str.replace(',', '.').astype(float))
                            pression = merged_df_cleaned[col_pressure].str.replace(',', '.').astype(float)
                        #cols_pressure_cleaned.append(col_pressure_cleaned)
                    #print(type(col_bird_cleaned))
                            Y+=oiseau.tolist()
                            X+=pression.tolist()
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y)
        model = LinearRegression().fit(X, Y)

        #merged_df_cleaned = merged_df.dropna(subset=['France_rendements_x', 'France_rendements_y'])
        # Afficher les coefficients et l'intercept
        print("Pour l'oiseau numero ",bird_file.replace("BIRD_","").replace("_0_counts_with_rendements_transformed.csv",""))
        print("Pour la pression",pressure_file.replace("_with_rendements","").replace(".csv",""))
        print("coefficient de la pression :",model.coef_)
        print("Intercept:", model.intercept_)