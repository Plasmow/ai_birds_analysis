import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# Charger les fichiers CSV
#farm_df = pd.read_csv('Pressure_with_rendements/FARMSIZE_with_rendements.csv', delimiter=';')
#bird_df = pd.read_csv('BirdCounts_with_rendements_transformed/BIRD_10010_0_counts_with_rendements_transformed.csv', delimiter=';')
bird_folder = 'BirdCounts_with_rendements_transformed'
pressure_folder = 'Pressure_with_rendements'
'''for pressure_file in os.listdir(pressure_folder):
    pressure_path = os.path.join(pressure_folder, pressure_file)  # Chemin complet du fichier
    pressure_df = pd.read_csv(pressure_path, delimiter=';')'''
for bird_file in os.listdir(bird_folder):
    bird_path = os.path.join(bird_folder, bird_file)  # Chemin complet du fichier
    bird_df = pd.read_csv(bird_path, delimiter=';')
    pressure_files = [pressure_file for pressure_file in os.listdir(pressure_folder)]
    pressure_files2 = [pressure_file.replace("_with_rendements","").replace(".csv","") for pressure_file in pressure_files]
    print(pressure_files2)
    pressure_path_tab = [os.path.join(pressure_folder, pressure_file) for pressure_file in pressure_files]
    pressure_dfs = [pd.read_csv(pressure_path, delimiter=';')  for pressure_path in pressure_path_tab]
    # Renommer les colonnes
    #farm_df.rename(columns={'Austria_rendements_FAR': 'Austria_rendements'}, inplace=True)

    # Fusionner les dataframes sur la colonne commune (Year)
    #merged_df = pressure_df.merge(bird_df, left_on='year', right_on='year')
    merged_df = bird_df.copy()
    for n in range(9):
        pressure_df = pressure_dfs[n]
        pressure_file = pressure_files2[n]
        merged_df = merged_df.merge(pressure_df, on='year', how='inner',suffixes=("","_"+pressure_file))
    #print(list(merged_df.columns))

    # Vérifier les noms des colonnes pour s'assurer qu'ils sont corrects
    #print(merged_df.columns)

    # Extraire les colonnes pertinentes pour la régression
    #X = merged_df[['France_rendements_y']]
    #y = merged_df['France_rendements_x']

    # Convertir les colonnes en numérique, remplacer les virgules par des points et gérer les valeurs manquantes
    #X = X.apply(lambda x: x.str.replace(',', '.')).astype(float)
    #y = y.str.replace(',', '.').astype(float)

    # Supprimer les lignes avec des valeurs NaN ou infinies
    #X = X.replace([np.inf, -np.inf], np.nan).dropna()
    #y = y.replace([np.inf, -np.inf], np.nan).dropna()

    # S'assurer que les longueurs de X et y sont cohérentes
    X = np.empty((0, 9))
    Y = []
    for col_bird in merged_df.columns:
        if col_bird.endswith("rendements"):
            cols_pressure = []
            for pressure_file in pressure_files2:
                for col_pressure in merged_df.columns:
                    if col_pressure == col_bird+"_" + pressure_file:
                        cols_pressure.append(col_pressure)
            #print(cols_pressure)
            if len(cols_pressure) == 9:   
                merged_df_cleaned = merged_df.loc[:, [col_bird]+cols_pressure]
                #merged_df_cleaned = merged_df.loc[:, [col_bird] + cols_pressure]
                #print(merged_df_cleaned)
                merged_df_cleaned = merged_df_cleaned.dropna()
                merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "0,0").all(axis=1)]
                merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "inf").all(axis=1)]
                
                #print(merged_df_cleaned[col_bird])
                if not merged_df_cleaned.empty:
                    ''''col_bird_cleaned = merged_df_cleaned[[col_bird]].apply(lambda x: x.str.replace(',', '.')).astype(float)
                    col_pressure_cleaned = merged_df_cleaned[[col_pressure]].apply(lambda x: x.str.replace(',', '.')).astype(float)'''
                    oiseau = merged_df_cleaned[col_bird].str.replace(',', '.').astype(float)
                    #cols_pressure_cleaned = []
                    #for col_pressure in cols_pressure:    #[col for col in merged_df_cleaned.columns if col != col_bird]:
                        #if col_pressure == col_bird+"_" + pressure_file:
                            #print(col_pressure)
                            #print(merged_df_cleaned[col_pressure])
                            #col_pressure_cleaned = merged_df_cleaned[[col_pressure]].apply(lambda x: x.str.replace(',', '.')).astype(float)
                            #print(merged_df_cleaned[col_pressure])
                            #print(merged_df_cleaned[col_pressure].str.replace(',', '.').astype(float))
                    pressions = merged_df_cleaned[cols_pressure].applymap(lambda x: float(str(x).replace(',', '.')))
                        #cols_pressure_cleaned.append(col_pressure_cleaned)
                    #print(type(col_bird_cleaned))
                    Y+=oiseau.tolist()
                    pressions_array = pressions.to_numpy()
                    X = np.vstack((X,pressions_array))
    #print(len(X),len(Y))
    #Y_res = np.array(Y)
    #X_res = np.array(X)
    #print(X_res)
    #print(Y_res)
    #print(len(X),len(Y))


    model = LinearRegression().fit(X, Y)

    #merged_df_cleaned = merged_df.dropna(subset=['France_rendements_x', 'France_rendements_y'])
    '''
    X_cleaned = merged_df_cleaned[['France_rendements_x']].apply(lambda x: x.str.replace(',', '.')).astype(float)
    Y_cleaned = merged_df_cleaned['France_rendements_y'].str.replace(',', '.').astype(float)

    # Créer le modèle et l'ajuster
    model = LinearRegression().fit(X_cleaned, Y_cleaned)
    '''
    # Afficher les coefficients et l'intercept
    print("Pour l'oiseau numero ",bird_file.replace("BIRD_","").replace("_0_counts_with_rendements_transformed.csv",""))
    for i in range(9):
        print("coefficient de la pression",pressure_files2[i],":",model.coef_[i])
    print("Intercept:", model.intercept_)