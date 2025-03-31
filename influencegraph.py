import pandas as pd
import os


#The goal is to create a dictionary for a specific date of dataframes, where the keys are the countries
#and the values are tuples of the form (pressure_type,value)

#This allows us to plot birds at time t as a function of pressure at time tthen birds at time t+1 as a function of pressure at time t

def extract_data_for_date_from_folder(folder_path, date):
    """
    Parcourt tous les fichiers CSV dans un dossier, extrait les données pour une date donnée,
    et stocke les valeurs dans un dictionnaire avec les noms des colonnes comme clés.
    Les listes contiennent des tuples (3 premiers caractères du nom du fichier, valeur).

    :param folder_path: Chemin du dossier contenant les fichiers CSV.
    :param date: Date pour laquelle extraire les données (format identique à celui des fichiers CSV).
    :return: Un dictionnaire avec les noms des colonnes comme clés et des listes de tuples (3 premiers caractères du nom du fichier, valeur).
    """
    data_dict = {}

    # Parcourir tous les fichiers dans le dossier
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Vérifier que le fichier est un CSV
            file_path = os.path.join(folder_path, filename)

            # Charger le fichier CSV avec les en-têtes
            df = pd.read_csv(file_path, sep=";", decimal=",", header=0)

            # Définir la première colonne comme index
            df.set_index(df.columns[0], inplace=True)

            # Vérifier si la date existe dans l'index
            if date in df.index:
                # Extraire les données pour la date donnée
                row = df.loc[date]

                # Ajouter les valeurs de la ligne au dictionnaire
                for col_name, value in row.items():
                    if col_name not in data_dict:
                        data_dict[col_name] = []  # Initialiser une liste pour chaque colonne
                    # Ajouter un tuple (3 premiers caractères du nom du fichier, valeur)
                    data_dict[col_name].append((filename[:3].capitalize(), value))

    return data_dict

# Exemple d'utilisation
folder_path = "Pressure_with_rendements"  # Chemin vers le dossier contenant les fichiers CSV
date_t = 1980  # Remplacez par la date souhaitée

data_for_date = extract_data_for_date_from_folder(folder_path, date_t)
print(f"Données pour la date {date_t} :\n{data_for_date}")




