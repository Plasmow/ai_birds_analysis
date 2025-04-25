import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Charger les fichiers CSV
farm_df = pd.read_csv('Pressure_with_rendements/FARMSIZE_with_rendements.csv', delimiter=';')
bird_df = pd.read_csv('BirdCounts_with_rendements_transformed/BIRD_90_0_counts_with_rendements_transformed.csv', delimiter=';')

# Renommer les colonnes
#farm_df.rename(columns={'Austria_rendements_FAR': 'Austria_rendements'}, inplace=True)

# Fusionner les dataframes sur la colonne commune (Year)
merged_df = farm_df.merge(bird_df, left_on='year', right_on='year')

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

X = []
Y = []
for col_bird in merged_df.columns:
    if col_bird.endswith("rendements_x"):
        for col_farm in merged_df.columns:
            if col_farm == col_bird.replace("_x","_y"):
                merged_df_cleaned = merged_df.dropna(subset=[col_bird, col_farm])
                merged_df_cleaned = merged_df_cleaned[(merged_df_cleaned[col_bird] != "0.0") & (merged_df_cleaned[col_bird] != "inf") & (merged_df_cleaned[col_farm] != "0.0") & (merged_df_cleaned[col_farm] != "inf")]
                ''''col_bird_cleaned = merged_df_cleaned[[col_bird]].apply(lambda x: x.str.replace(',', '.')).astype(float)
                col_farm_cleaned = merged_df_cleaned[[col_farm]].apply(lambda x: x.str.replace(',', '.')).astype(float)'''
                col_bird_cleaned = merged_df_cleaned[col_bird].str.replace(',', '.').astype(float)
                col_farm_cleaned = merged_df_cleaned[col_farm].str.replace(',', '.').astype(float)
                print(type(col_bird_cleaned))
                X+=col_bird_cleaned.tolist()
                Y+=col_farm_cleaned.tolist()
print(len(X),len(Y))
Y_res = np.array(X)
X_res = np.array(Y).reshape(-1, 1)
print(X_res)
print(Y_res)


model = LinearRegression().fit(X_res, Y_res)

#merged_df_cleaned = merged_df.dropna(subset=['France_rendements_x', 'France_rendements_y'])
'''
X_cleaned = merged_df_cleaned[['France_rendements_x']].apply(lambda x: x.str.replace(',', '.')).astype(float)
Y_cleaned = merged_df_cleaned['France_rendements_y'].str.replace(',', '.').astype(float)

# Créer le modèle et l'ajuster
model = LinearRegression().fit(X_cleaned, Y_cleaned)
'''
# Afficher les coefficients et l'intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)