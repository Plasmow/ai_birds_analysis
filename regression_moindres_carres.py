import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

# Initialisation des structures pour la régression globale
all_Y = []
all_X = np.empty((0, 8))
lst_pressure = [[] for _ in range(8)]

# Charger les fichiers CSV
bird_folder = 'BirdCounts_with_rendements_transformed'
pressure_folder = 'Pressure_with_rendements'
for bird_file in os.listdir(bird_folder):
    bird_path = os.path.join(bird_folder, bird_file)  
    bird_df = pd.read_csv(bird_path, delimiter=';')
    pressure_files = [pressure_file for pressure_file in os.listdir(pressure_folder)]
    pressure_files2 = [pressure_file.replace("_with_rendements","").replace(".csv","") for pressure_file in pressure_files]
    pressure_path_tab = [os.path.join(pressure_folder, pressure_file) for pressure_file in pressure_files]
    pressure_dfs = [pd.read_csv(pressure_path, delimiter=';')  for pressure_path in pressure_path_tab]
    
    # Fusion des pressions avec les données oiseaux
    merged_df = bird_df.copy()
    for n in range(8):
        pressure_df = pressure_dfs[n]
        pressure_file = pressure_files2[n]
        merged_df = merged_df.merge(pressure_df, on='year', how='inner',suffixes=("","_"+pressure_file))

    # Préparation des données pour la régression espèce par espèce
    X = np.empty((0, 8))
    Y = []

    for col_bird in merged_df.columns:
        if col_bird.endswith("rendements"):
            cols_pressure = []
            for pressure_file in pressure_files2:
                for col_pressure in merged_df.columns:
                    if col_pressure == col_bird+"_" + pressure_file:
                        cols_pressure.append(col_pressure)
            if len(cols_pressure) == 8:   
                merged_df_cleaned = merged_df.loc[:, [col_bird]+cols_pressure]
                merged_df_cleaned = merged_df_cleaned.dropna()
                merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "0,0").all(axis=1)]
                merged_df_cleaned = merged_df_cleaned[merged_df_cleaned.applymap(lambda x: x != "inf").all(axis=1)]
                if not merged_df_cleaned.empty:
                    oiseau = merged_df_cleaned[col_bird].str.replace(',', '.').astype(float)
                    pressions = merged_df_cleaned[cols_pressure].applymap(lambda x: float(str(x).replace(',', '.')))
                    Y+=oiseau.tolist()
                    pressions_array = pressions.to_numpy()
                    X = np.vstack((X,pressions_array))

    # Régression pour chaque oiseau si données présentes
    if not(len(Y)==0):
        print(len(Y))
        all_Y += Y
        all_X = np.vstack((all_X,X))
        model = LinearRegression().fit(X, Y)

    
        # Afficher les coefficients et l'intercept
        print("Pour l'oiseau numero ",bird_file.replace("BIRD_","").replace("_0_counts_with_rendements_transformed.csv",""))
        for i in range(8):
            print("coefficient de la pression",pressure_files2[i],":",model.coef_[i])
            lst_pressure[i].append(model.coef_[i])
        print("Intercept:", model.intercept_)
        r2_score = model.score(X, Y)
        print("R² (coefficient de détermination) :", r2_score)

# Régression globale sur tous les oiseaux
all_model = LinearRegression().fit(all_X, all_Y)
print("Pour l'ensemble des oiseaux")
for i in range(8):
    print("coefficient de la pression",pressure_files2[i],":",all_model.coef_[i])
print("Intercept:", model.intercept_)
r2_score = model.score(X, Y)
print("R² (coefficient de détermination) :", r2_score)

# Coefficients pour l'ensemble des oiseaux
'''
plt.bar(pressure_files2, all_model.coef_)
plt.xlabel("Pressions")
plt.ylabel("Coefficients")
plt.title("Coefficients de régression pour l'ensemble des oiseaux")
plt.xticks(rotation=45)
plt.show()
'''

# Histogramme de fréquence des coefficients de la première pression
first_line = lst_pressure[0]
plt.hist(first_line, bins=10, edgecolor='black')  # bins=10 pour 10 intervalles
plt.xlabel("Valeurs des coefficients")
plt.ylabel("Fréquence")
plt.title("Histogramme de fréquence des coefficients (1ère pression)")
plt.show()