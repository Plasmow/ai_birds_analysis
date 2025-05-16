import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Initialisation des dossiers et des matrices globales pour la régression sur tous les oiseaux
bird_folder = 'BirdCounts_with_rendements_transformed'
pressure_folder = 'Pressure_with_rendements'
all_Y = np.empty((0, 1))
all_X = np.empty((0, 8))

# Parcours de chaque fichier d'oiseau
for bird_file in os.listdir(bird_folder):
    bird_path = os.path.join(bird_folder, bird_file) 
    bird_df = pd.read_csv(bird_path, delimiter=';')
    pressure_files = [pressure_file for pressure_file in os.listdir(pressure_folder)]
    pressure_files2 = [pressure_file.replace("_with_rendements", "").replace(".csv", "") for pressure_file in pressure_files]
    pressure_path_tab = [os.path.join(pressure_folder, pressure_file) for pressure_file in pressure_files]
    pressure_dfs = [pd.read_csv(pressure_path, delimiter=';') for pressure_path in pressure_path_tab]

    # Fusion des pressions avec les données oiseaux
    merged_df = bird_df.copy()
    for n in range(8):
        pressure_df = pressure_dfs[n]
        pressure_file = pressure_files2[n]
        merged_df = merged_df.merge(pressure_df, on='year', how='inner', suffixes=("", "_" + pressure_file))

    # Préparation des matrices pour la régression de chaque oiseau
    X = np.empty((0, 8))
    Y = []

    # Recherche des colonnes de rendements et des pressions associées
    for col_bird in merged_df.columns:
        if col_bird.endswith("rendements"):
            cols_pressure = []
            for pressure_file in pressure_files2:
                for col_pressure in merged_df.columns:
                    if col_pressure == col_bird + "_" + pressure_file:
                        cols_pressure.append(col_pressure)

            # Si on a bien 8 pressions associées
            if len(cols_pressure) == 8:
                merged_df_cleaned = merged_df.loc[:, [col_bird] + cols_pressure]
                # Nettoyage des données (suppression des NaN, "0,0" et "inf")
                merged_df_cleaned = merged_df_cleaned.dropna()
                merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "0,0").all(axis=1)]
                merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "inf").all(axis=1)]

                if not merged_df_cleaned.empty:
                    # Conversion des chaînes en float
                    oiseau = merged_df_cleaned[col_bird].str.replace(',', '.').astype(float)
                    pressions = merged_df_cleaned[cols_pressure].applymap(lambda x: float(str(x).replace(',', '.')))
                    pressions_array = pressions.to_numpy()

                    # Ajout des données à X et Y
                    Y += oiseau.tolist()
                    X = np.vstack((X, pressions_array))

    # Régression Lasso pour chaque oiseau si données présentes
    if not(len(Y)==0):
        X = np.array(X)
        Y = np.array(Y)
        Y = np.array(Y).reshape(-1, 1)  
        all_Y = np.vstack((all_Y, Y))
        all_X = np.vstack((all_X,X))
        model = Lasso(alpha=0.001).fit(X, Y)  # `alpha` contrôle la pénalisation L1

        # Afficher les coefficients et l'intercept
        print("Pour l'oiseau numero ", bird_file.replace("BIRD_", "").replace("_0_counts_with_rendements_transformed.csv", ""))
        for i in range(8):
            print("coefficient de la pression", pressure_files2[i], ":", model.coef_[i])
        print("Intercept:", model.intercept_)

# Régression Lasso globale sur tous les oiseaux
all_model = Lasso(alpha=0.001).fit(all_X, all_Y)  # `alpha` contrôle la pénalisation L1
print("Pour l'ensemble des oiseaux")
for i in range(8):
    print("coefficient de la pression",pressure_files2[i],":",all_model.coef_[i])
print("Intercept:", model.intercept_)

# Visualisation des coefficients globaux
plt.bar(pressure_files2, all_model.coef_)
plt.xlabel("Pressions")
plt.ylabel("Coefficients")
plt.title("Coefficients de régression pour l'ensemble des oiseaux")
plt.xticks(rotation=45)
plt.show()
